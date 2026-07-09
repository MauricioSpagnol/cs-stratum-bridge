//! The OPoI lifecycle engine: polls for on-chain pending requests, assigns
//! them to connected miners, receives their answers, and drives
//! commit -> reveal -> publish. This is the piece that replaces back-pool's
//! `opoiPool.js`, with the assignment-verification and startup-recovery
//! fixes the audit called for baked in from the start rather than bolted on.
//!
//! Important: `self.opoi_address` (the bridge's own OPoI stake identity) is
//! what's passed as `miner_address` to every commit/reveal RPC — the bridge
//! custodies exactly one on-chain stake and commits/reveals as itself,
//! regardless of which downstream miner actually produced the answer. The
//! per-connection wallet is tracked separately, purely for payout
//! attribution (see src/payout).

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use sqlx::PgPool;

use crate::db;
use crate::db::models::Submission;
use crate::error::AppError;
use crate::miner_registry::MinerRegistry;
use crate::opoi::assignment::AssignmentTracker;
use crate::opoi::handler::OpoiHandler;
use crate::opoi::wire::{build_assign_line, OpoiAssign, OpoiSubmitResultParams};
use crate::rpc::types::OpoiRequest;
use crate::rpc::CsdRpcClient;

struct PendingRequest {
    request: OpoiRequest,
    prompt_hex: Option<String>,
}

/// Read-only view of one cached pending request, for the diagnostic
/// `GET /cscoin/opoi/pending` HTTP endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PendingSnapshotEntry {
    pub request_id: String,
    pub model: String,
    pub has_prompt: bool,
    pub assigned_to: Option<String>,
}

pub struct OpoiEngine {
    csd: Arc<CsdRpcClient>,
    db: PgPool,
    registry: Arc<MinerRegistry>,
    assignments: AssignmentTracker,
    pending: DashMap<String, PendingRequest>,
    last_assigned: Mutex<Option<String>>,
    opoi_address: String,
}

impl OpoiEngine {
    pub fn new(csd: Arc<CsdRpcClient>, db: PgPool, registry: Arc<MinerRegistry>, opoi_address: String) -> Self {
        Self {
            csd,
            db,
            registry,
            assignments: AssignmentTracker::new(),
            pending: DashMap::new(),
            last_assigned: Mutex::new(None),
            opoi_address,
        }
    }

    /// The bridge's own OPoI stake address (used by the payout module to
    /// know which wallet's income is being distributed).
    pub fn opoi_address(&self) -> &str {
        &self.opoi_address
    }

    pub fn db_pool(&self) -> &PgPool {
        &self.db
    }

    pub fn csd_client(&self) -> Arc<CsdRpcClient> {
        self.csd.clone()
    }

    /// Ensures the bridge has an ACTIVE OPoI stake: stakes if none exists
    /// (requires collateral to be configured), renews if SUSPENDED. Called
    /// once at startup, before anything else runs.
    pub async fn ensure_stake(
        &self,
        collateral_txid: &str,
        collateral_vout: u32,
        endpoint: &str,
        model_id: &str,
    ) -> anyhow::Result<()> {
        match self.csd.get_opoi_stake(&self.opoi_address).await? {
            None => {
                if collateral_txid.is_empty() {
                    anyhow::bail!(
                        "no OPoI stake found for {} and no OPOI_COLLATERAL_TXID configured to create one",
                        self.opoi_address
                    );
                }
                let res = self
                    .csd
                    .stake_opoi(&self.opoi_address, collateral_txid, collateral_vout, endpoint, model_id)
                    .await?;
                let _ = db::repo::log_stake_event(
                    &self.db,
                    "STAKE",
                    Some(&res.txid),
                    &format!("initial stake, amount={}", res.amount),
                )
                .await;
                tracing::info!(txid = %res.txid, "OPoI stake created");
            }
            Some(stake) if stake.status == "SUSPENDED" => {
                tracing::warn!("OPoI stake is SUSPENDED at startup; renewing immediately");
                self.renew_tick().await;
            }
            Some(stake) => {
                tracing::info!(status = %stake.status, "OPoI stake already exists");
            }
        }
        Ok(())
    }

    /// Periodic keep-alive renewal. Called on an interval by main.rs, and
    /// once at startup by `ensure_stake` if the stake was found SUSPENDED.
    pub async fn renew_tick(&self) {
        match self.csd.renew_opoi_stake(&self.opoi_address).await {
            Ok(txid) => {
                let _ = db::repo::log_stake_event(&self.db, "RENEW", Some(&txid), "periodic renewal").await;
                tracing::info!(txid = %txid, "OPoI stake renewed");
            }
            Err(e) => {
                let _ = db::repo::log_stake_event(&self.db, "RENEW_ERROR", None, &e.to_string()).await;
                tracing::warn!(error = %e, "OPoI stake renewal failed");
            }
        }
    }

    /// Startup recovery (fix for the audit's "no restart recovery" finding):
    /// COMMITTED/REVEALED rows already self-heal via the normal poll loop
    /// starting again — the only genuinely orphaned state is RECEIVED, since
    /// nothing else ever re-drives it. Must be called before the normal
    /// loops start.
    pub async fn recover_on_startup(&self) -> anyhow::Result<()> {
        let received = db::repo::list_received(&self.db).await?;
        tracing::info!(count = received.len(), "recovering RECEIVED submissions left over from a prior run");

        stream::iter(received)
            .for_each_concurrent(4, |sub| {
                let csd = self.csd.clone();
                let pool = self.db.clone();
                let opoi_address = self.opoi_address.clone();
                async move {
                    do_commit(csd, pool, opoi_address, sub.id, sub.request_id, sub.response_hash, sub.token_count as u32).await;
                }
            })
            .await;

        let committed = db::repo::list_committed(&self.db).await?.len();
        let revealed = db::repo::list_revealed_unpublished(&self.db).await?.len();
        tracing::info!(committed, revealed, "these self-heal via the normal reveal/publish poll loop, no action needed");
        Ok(())
    }

    /// Called by the HTTP prompt-intake endpoint. Stashes the plaintext
    /// prompt against a cached (or freshly-fetched) on-chain request so it
    /// becomes eligible for assignment. Returns false if the request_id is
    /// unknown even to the daemon.
    pub async fn receive_prompt(&self, request_id: &str, prompt_hex: String) -> anyhow::Result<bool> {
        if let Some(mut entry) = self.pending.get_mut(request_id) {
            entry.prompt_hex = Some(prompt_hex);
            return Ok(true);
        }
        match self.csd.get_request(request_id).await {
            Ok(request) => {
                self.pending.insert(request_id.to_string(), PendingRequest { request, prompt_hex: Some(prompt_hex) });
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    /// Diagnostic snapshot of everything currently cached in memory (does
    /// not touch the DB or csd) — backs `GET /cscoin/opoi/pending`.
    pub fn pending_snapshot(&self) -> Vec<PendingSnapshotEntry> {
        self.pending
            .iter()
            .map(|e| PendingSnapshotEntry {
                request_id: e.key().clone(),
                model: e.request.model.clone(),
                has_prompt: e.prompt_hex.is_some(),
                assigned_to: self.assignments.assignee(e.key()),
            })
            .collect()
    }

    /// One tick of: poll csd for PENDING requests (caching any new ones),
    /// then assign every request that has both a known prompt and no
    /// current assignee to the next connected miner (round-robin).
    pub async fn poll_and_assign_tick(&self) -> anyhow::Result<()> {
        // Learned from live testing (not from the original design): csd
        // keeps reporting a request as PENDING via listopoirequests even
        // after its COMMIT is confirmed on-chain (status only changes once
        // REVEALED, if then) — without this check, an already-answered
        // request would be re-inserted as a fresh "unassigned, no prompt"
        // ghost entry on every single poll tick, forever, since a plain
        // or_insert_with can't tell "never seen" apart from "already
        // resolved, csd just hasn't updated its status field for it".
        for req in self.csd.list_pending_requests().await? {
            if self.pending.contains_key(&req.request_id) {
                continue;
            }
            if db::repo::find_active_by_request_id(&self.db, &req.request_id).await?.is_some() {
                continue; // already has a submission in flight; not a fresh assignment candidate
            }
            self.pending
                .entry(req.request_id.clone())
                .or_insert_with(|| PendingRequest { request: req, prompt_hex: None });
        }

        let candidates: Vec<String> = self
            .pending
            .iter()
            .filter(|e| e.prompt_hex.is_some() && self.assignments.assignee(e.key()).is_none())
            .map(|e| e.key().clone())
            .collect();

        for request_id in candidates {
            let Some((prompt_hex, model, max_tokens, task_type, task_class)) = self.pending.get(&request_id).map(|e| {
                (
                    e.prompt_hex.clone().unwrap(),
                    e.request.model.clone(),
                    e.request.max_tokens,
                    e.request.task_type.clone(),
                    e.request.task_class.clone(),
                )
            }) else {
                continue;
            };

            let wallet = {
                let mut last = self.last_assigned.lock();
                let Some(wallet) = self.registry.pick_next(&last) else {
                    break; // no miners connected right now, nothing more to do this tick
                };
                *last = Some(wallet.clone());
                wallet
            };

            if self.assignments.try_assign(&request_id, &wallet) {
                if let Some(tx) = self.registry.get(&wallet) {
                    let assign = OpoiAssign { request_id: request_id.clone(), model, prompt_hex, max_tokens, task_type, task_class };
                    let _ = tx.send(build_assign_line(&assign));
                    tracing::info!(request_id = %request_id, wallet = %wallet, "assigned OPoI request");
                } else {
                    // Registry entry vanished between pick_next and get (miner
                    // disconnected in that instant) — release immediately so
                    // the next tick can hand it to someone else.
                    self.assignments.remove(&request_id);
                }
            }
        }

        Ok(())
    }

    /// One tick of: reveal every COMMITTED submission whose window has
    /// closed, then publish every REVEALED submission not yet published.
    pub async fn poll_reveal_tick(&self) -> anyhow::Result<()> {
        let height = self.csd.get_chain_height().await?;

        for sub in db::repo::list_committed(&self.db).await? {
            if let Some(closes) = sub.commit_window_closes_at_height {
                if height >= closes as u64 {
                    self.do_reveal(&sub).await;
                }
            }
        }

        for sub in db::repo::list_revealed_unpublished(&self.db).await? {
            self.do_publish(&sub).await;
        }

        Ok(())
    }

    async fn do_reveal(&self, sub: &Submission) {
        let Some(nonce_hex) = sub.commit_nonce_hex.clone() else {
            tracing::error!(submission_id = sub.id, "COMMITTED submission missing commit_nonce_hex; cannot reveal");
            return;
        };

        match self
            .csd
            .submit_response_reveal(&sub.request_id, &sub.response_hash, &self.opoi_address, sub.token_count as u32, &nonce_hex)
            .await
        {
            Ok(txid) => {
                if let Err(e) = db::repo::mark_revealed(&self.db, sub.id, &txid).await {
                    tracing::error!(error = %e, submission_id = sub.id, "failed to persist REVEALED state");
                } else {
                    tracing::info!(submission_id = sub.id, request_id = %sub.request_id, txid = %txid, "OPoI response revealed on-chain");
                }
            }
            // Deliberate retry-forever: stays COMMITTED, tried again next tick.
            Err(e) => {
                tracing::debug!(error = %e, submission_id = sub.id, "reveal not ready yet / failed, will retry");
            }
        }
    }

    async fn do_publish(&self, sub: &Submission) {
        let Some(response_hex) = sub.response_hex.clone() else {
            tracing::error!(submission_id = sub.id, "REVEALED submission missing response_hex; cannot publish");
            return;
        };

        match self.csd.submit_content(&sub.request_id, "RESPONSE", &response_hex).await {
            Ok(()) => {
                // Fetched fresh (not from any submit-time cache) so the
                // reward figure is correct even after a restart.
                let reward = match self.csd.get_request(&sub.request_id).await {
                    Ok(req) => req.payment + (sub.token_count as f64) * req.fee_per_token,
                    Err(e) => {
                        tracing::warn!(
                            error = %e, submission_id = sub.id,
                            "could not fetch request to compute reward_amount; recording 0.0, will need manual correction"
                        );
                        0.0
                    }
                };
                if let Err(e) = db::repo::mark_published(&self.db, sub.id, reward).await {
                    tracing::error!(error = %e, submission_id = sub.id, "failed to persist PUBLISHED state");
                } else {
                    tracing::info!(submission_id = sub.id, request_id = %sub.request_id, reward_amount = reward, "OPoI content published, reward recorded");
                }
            }
            // Deliberate retry-forever: stays REVEALED, tried again next tick.
            Err(e) => {
                tracing::debug!(error = %e, submission_id = sub.id, "content publish not ready yet / failed, will retry");
            }
        }
    }
}

#[async_trait]
impl OpoiHandler for OpoiEngine {
    async fn handle_submit_result(&self, wallet: &str, params: OpoiSubmitResultParams) -> Result<i64, AppError> {
        // 1. Assignment verification — the actual fix for the audit's
        // "any miner can answer any request_id" finding.
        match self.assignments.assignee(&params.request_id) {
            None => return Err(AppError::UnknownRequest),
            Some(assigned) if assigned != wallet => return Err(AppError::NotAssignedToCaller),
            _ => {}
        }

        // 2. Defense-in-depth duplicate guard (also backed by the DB's
        // partial unique index).
        if db::repo::find_active_by_request_id(&self.db, &params.request_id).await?.is_some() {
            return Err(AppError::Rejected("request_id already has an active submission".into()));
        }

        // 3. Verify the claimed hash actually matches the claimed content —
        // avoids burning a real on-chain COMMIT on a mismatched pair that
        // could never publish later.
        let raw = hex::decode(&params.response_hex)
            .map_err(|_| AppError::Rejected("response_hex is not valid hex".into()))?;
        let computed = hex::encode(Sha256::digest(&raw));
        if !computed.eq_ignore_ascii_case(&params.response_hash) {
            return Err(AppError::Rejected("response_hash does not match sha256(response_hex)".into()));
        }

        // 4. Best-effort bookkeeping fields from the cache (informational only).
        let (model, prompt_hash) = self
            .pending
            .get(&params.request_id)
            .map(|e| (Some(e.request.model.clone()), Some(e.request.prompt_hash.clone())))
            .unwrap_or((None, None));

        // 5. Persist.
        let id = db::repo::create_submission(
            &self.db,
            wallet,
            &params.request_id,
            model.as_deref(),
            prompt_hash.as_deref(),
            None,
            None,
            &params.response_hash,
            &params.response_hex,
            params.token_count as i32,
        )
        .await?;

        // The request is now spoken for — never reassignable again.
        self.assignments.remove(&params.request_id);
        self.pending.remove(&params.request_id);

        // 6. Fire-and-forget commit (mirrors the original design: the
        // miner's ACK doesn't wait on the on-chain RPC).
        let csd = self.csd.clone();
        let pool = self.db.clone();
        let opoi_address = self.opoi_address.clone();
        let request_id = params.request_id.clone();
        let response_hash = params.response_hash.clone();
        let token_count = params.token_count;
        tokio::spawn(async move {
            do_commit(csd, pool, opoi_address, id, request_id, response_hash, token_count).await;
        });

        Ok(id)
    }

    async fn on_disconnect(&self, wallet: &str) {
        self.assignments.release_for_wallet(wallet);
    }
}

/// Free function (not a `&self` method) so it can be called both from a
/// `tokio::spawn`ed 'static task (the normal fire-and-forget path) and from
/// `recover_on_startup`'s bounded-concurrency stream — neither needs to
/// borrow `OpoiEngine` itself, only these cheaply-cloneable pieces.
async fn do_commit(
    csd: Arc<CsdRpcClient>,
    pool: PgPool,
    opoi_address: String,
    submission_id: i64,
    request_id: String,
    response_hash: String,
    token_count: u32,
) {
    match csd.submit_response_commit(&request_id, &response_hash, &opoi_address, token_count).await {
        Ok(res) => {
            if let Err(e) = db::repo::mark_committed(&pool, submission_id, &res.txid, &res.nonce_hex, res.commit_window_closes_at_height as i32).await {
                tracing::error!(error = %e, submission_id, "failed to persist COMMITTED state after successful RPC");
            } else {
                tracing::info!(submission_id, request_id = %request_id, txid = %res.txid, "OPoI response committed on-chain");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, submission_id, request_id = %request_id, "submitopoiresponsecommit failed; marking FAILED");
            if let Err(e2) = db::repo::mark_failed(&pool, submission_id, &e.to_string()).await {
                tracing::error!(error = %e2, submission_id, "failed to persist FAILED state");
            }
        }
    }
}
