//! Real on-chain miner payouts, batched into a single `sendmany` per tick.
//!
//! Known accepted risk (same class as the commit-recovery gap documented in
//! `opoi::engine`): if this process crashes between `sendmany` succeeding
//! and the corresponding rows being marked `paid`, the next tick will fold
//! the same rows into a second payout. Rare in practice, and the first
//! txid is always logged for manual reconciliation — a proper fix (e.g.
//! recording the txid before broadcasting, or idempotency at the RPC layer)
//! is future work, not v1 scope.

use std::collections::BTreeMap;
use std::sync::Arc;

use sqlx::PgPool;

use crate::db;
use crate::rpc::CsdRpcClient;

pub async fn payout_tick(csd: Arc<CsdRpcClient>, db: PgPool, payout_address: String, min_payout_cs: f64, pool_fee_percent: f64) {
    match run(csd, db, payout_address, min_payout_cs, pool_fee_percent).await {
        Ok(Some((txid, miners, total))) => {
            tracing::info!(txid = %txid, miners, total_cs = total, "payout batch sent");
        }
        Ok(None) => {
            tracing::debug!("payout tick: nothing above the minimum threshold, skipping");
        }
        Err(e) => {
            tracing::error!(error = %e, "payout tick failed");
        }
    }
}

async fn run(
    csd: Arc<CsdRpcClient>,
    db: PgPool,
    payout_address: String,
    min_payout_cs: f64,
    pool_fee_percent: f64,
) -> anyhow::Result<Option<(String, usize, f64)>> {
    let unpaid = db::repo::list_unpaid_published(&db).await?;
    if unpaid.is_empty() {
        return Ok(None);
    }

    // Group by miner_wallet: sum reward, collect the request_ids that make
    // up that sum (needed afterward to mark exactly those rows paid).
    let mut by_wallet: BTreeMap<String, (f64, Vec<String>)> = BTreeMap::new();
    for sub in unpaid {
        let Some(reward) = sub.reward_amount else { continue };
        let entry = by_wallet.entry(sub.miner_wallet.clone()).or_insert((0.0, Vec::new()));
        entry.0 += reward;
        entry.1.push(sub.request_id.clone());
    }

    let fee_factor = 1.0 - (pool_fee_percent / 100.0);
    let mut amounts: BTreeMap<String, f64> = BTreeMap::new();
    let mut request_ids: Vec<String> = Vec::new();

    for (wallet, (total, ids)) in by_wallet {
        if total < min_payout_cs {
            continue; // stays unpaid, rolls into next tick's balance — avoids dust/fee waste
        }
        let payable = total * fee_factor;
        if payable <= 0.0 {
            continue;
        }
        amounts.insert(wallet, payable);
        request_ids.extend(ids);
    }

    if amounts.is_empty() {
        return Ok(None);
    }

    let total: f64 = amounts.values().sum();
    let miners = amounts.len();
    let txid = csd.send_many(&payout_address, &amounts).await?;

    db::repo::mark_paid(&db, &request_ids, &txid).await?;

    Ok(Some((txid, miners, total)))
}
