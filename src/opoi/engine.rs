//! The OPoI lifecycle engine: polls for on-chain pending requests, assigns
//! them to connected miners, receives their answers, and drives
//! commit -> reveal -> publish. This is the piece that replaces back-pool's
//! `opoiPool.js`, with the assignment-verification and startup-recovery
//! fixes the audit called for baked in from the start rather than bolted on.
//!
//! Important: the bridge custodies a POOL of on-chain stake identities
//! (`self.stake_pool`, see stake_pool.rs), not just one — commits/reveals
//! happen as one of them, regardless of which downstream miner actually
//! produced the answer. This exists because REVEAL is VRF-gated at ~3%
//! eligibility per address on mainnet/testnet, and REVEAL must use the same
//! address that COMMITted; `do_commit` therefore commits with EVERY pool
//! address in parallel, and `poll_reveal_tick` tries revealing with each
//! COMMITTED attempt (see `opoi_commit_attempts`) until one succeeds. The
//! per-connection miner wallet is tracked separately, purely for payout
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
use crate::stake_pool::StakePool;

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
    stake_pool: Arc<StakePool>,
}

impl OpoiEngine {
    pub fn new(csd: Arc<CsdRpcClient>, db: PgPool, registry: Arc<MinerRegistry>, stake_pool: Arc<StakePool>) -> Self {
        Self {
            csd,
            db,
            registry,
            assignments: AssignmentTracker::new(),
            pending: DashMap::new(),
            last_assigned: Mutex::new(None),
            stake_pool,
        }
    }

    /// The bridge's primary OPoI stake address (used for logging and as
    /// the payout fallback — see `Config::effective_payout_address`).
    pub fn primary_address(&self) -> &str {
        self.stake_pool.primary()
    }

    pub fn db_pool(&self) -> &PgPool {
        &self.db
    }

    pub fn csd_client(&self) -> Arc<CsdRpcClient> {
        self.csd.clone()
    }

    /// Ensures every pool address has an ACTIVE OPoI stake where possible:
    /// the PRIMARY address (`stake_pool.primary()`) is auto-staked from the
    /// configured collateral if missing, exactly as before the pool existed.
    /// Every other pool address must already have an ACTIVE stake created
    /// out-of-band (`stakeopoi`) — there's no per-address collateral config
    /// to auto-bootstrap them, so a missing one is logged loudly and
    /// skipped rather than failing startup (it just contributes no
    /// eligibility coverage until staked manually). Any SUSPENDED address,
    /// primary or not, is renewed immediately. Called once at startup,
    /// before anything else runs.
    pub async fn ensure_stake(
        &self,
        collateral_txid: &str,
        collateral_vout: u32,
        endpoint: &str,
        model_id: &str,
    ) -> anyhow::Result<()> {
        let primary = self.stake_pool.primary().to_string();
        match self.csd.get_opoi_stake(&primary).await? {
            None => {
                if collateral_txid.is_empty() {
                    anyhow::bail!(
                        "no OPoI stake found for primary address {} and no OPOI_COLLATERAL_TXID configured to create one",
                        primary
                    );
                }
                let res = self
                    .csd
                    .stake_opoi(&primary, collateral_txid, collateral_vout, endpoint, model_id)
                    .await?;
                let _ = db::repo::log_stake_event(
                    &self.db,
                    "STAKE",
                    Some(&res.txid),
                    &format!("initial stake for {primary}, amount={}", res.amount),
                )
                .await;
                tracing::info!(txid = %res.txid, address = %primary, "OPoI stake created (primary)");
            }
            Some(stake) if stake.status == "SUSPENDED" => {
                tracing::warn!(address = %primary, "OPoI stake is SUSPENDED at startup; renewing immediately");
                self.renew_one(&primary).await;
            }
            Some(stake) => {
                tracing::info!(address = %primary, status = %stake.status, "OPoI stake already exists (primary)");
            }
        }

        for addr in &self.stake_pool.addresses()[1..] {
            match self.csd.get_opoi_stake(addr).await? {
                None => {
                    tracing::error!(
                        address = %addr,
                        "OPoI pool address has NO stake — create one out-of-band via `stakeopoi` \
                         before this address contributes any eligibility coverage; skipping for now"
                    );
                }
                Some(stake) if stake.status == "SUSPENDED" => {
                    tracing::warn!(address = %addr, "OPoI pool address stake is SUSPENDED at startup; renewing immediately");
                    self.renew_one(addr).await;
                }
                Some(stake) => {
                    tracing::info!(address = %addr, status = %stake.status, "OPoI pool address stake already exists");
                }
            }
        }
        Ok(())
    }

    /// Periodic keep-alive renewal for every address in the pool. Called on
    /// an interval by main.rs, and once at startup by `ensure_stake` for any
    /// address found SUSPENDED.
    pub async fn renew_tick(&self) {
        for addr in self.stake_pool.addresses() {
            self.renew_one(addr).await;
        }
    }

    async fn renew_one(&self, address: &str) {
        match self.csd.renew_opoi_stake(address).await {
            Ok(txid) => {
                let _ = db::repo::log_stake_event(&self.db, "RENEW", Some(&txid), &format!("periodic renewal for {address}")).await;
                tracing::info!(txid = %txid, address = %address, "OPoI stake renewed");
            }
            Err(e) => {
                let _ = db::repo::log_stake_event(&self.db, "RENEW_ERROR", None, &format!("{address}: {e}")).await;
                tracing::warn!(error = %e, address = %address, "OPoI stake renewal failed");
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
                let stake_pool = self.stake_pool.clone();
                async move {
                    do_commit(csd, pool, stake_pool, sub.id, sub.request_id, sub.response_hash, sub.token_count as u32).await;
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

            // F15-H (Sessão 3): a request whose model has an ACTIVE manifest
            // is shard-routed — ShardEngine handles it (own poll tick, own
            // prompt cache), not the whole-model path here. Left cached in
            // `self.pending` rather than removed: harmless, re-checked next
            // tick at the same cost ShardEngine already pays for the same
            // model_id.
            if matches!(self.csd.get_model_manifest(&model).await, Ok(m) if m.status == "ACTIVE") {
                continue;
            }

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

    /// One tick of: for every submission still COMMITTED, try revealing
    /// with each pool address whose commit window has closed (stopping at
    /// the first VRF-eligible one), then publish every REVEALED submission
    /// not yet published.
    pub async fn poll_reveal_tick(&self) -> anyhow::Result<()> {
        let height = self.csd.get_chain_height().await?;

        let candidates = db::repo::list_reveal_ready_candidates(&self.db, height as i64).await?;
        let mut by_submission: std::collections::HashMap<i64, Vec<db::RevealCandidate>> = std::collections::HashMap::new();
        for c in candidates {
            by_submission.entry(c.submission_id).or_default().push(c);
        }
        for (_submission_id, attempts) in by_submission {
            self.try_reveal_any(attempts).await;
        }

        for sub in db::repo::list_revealed_unpublished(&self.db).await? {
            self.do_publish(&sub).await;
        }

        Ok(())
    }

    /// Tries revealing with each ready attempt in turn, stopping at the
    /// first one the daemon accepts (VRF-eligible for this address). The
    /// rest are left COMMITTED and simply never revealed once one succeeds
    /// — `poll_reveal_tick`'s next call excludes them anyway, since their
    /// parent submission will have flipped to REVEALED.
    async fn try_reveal_any(&self, candidates: Vec<db::RevealCandidate>) {
        for c in candidates {
            match self
                .csd
                .submit_response_reveal(&c.request_id, &c.response_hash, &c.opoi_address, c.token_count as u32, &c.nonce_hex)
                .await
            {
                Ok(txid) => {
                    if let Err(e) = db::repo::mark_attempt_revealed(&self.db, c.attempt_id, &txid).await {
                        tracing::error!(error = %e, attempt_id = c.attempt_id, "failed to persist REVEALED attempt");
                    }
                    if let Err(e) = db::repo::mark_revealed(&self.db, c.submission_id, &txid).await {
                        tracing::error!(error = %e, submission_id = c.submission_id, "failed to persist REVEALED submission");
                    } else {
                        tracing::info!(
                            submission_id = c.submission_id, request_id = %c.request_id,
                            address = %c.opoi_address, txid = %txid,
                            "OPoI response revealed on-chain (this pool address was VRF-eligible)"
                        );
                    }
                    return;
                }
                // Deliberate retry-forever across ticks: not eligible (or
                // transiently failed) this time, try the next candidate for
                // this submission; if none succeed this tick, all stay
                // COMMITTED and are re-tried next tick.
                Err(e) => {
                    tracing::debug!(
                        error = %e, submission_id = c.submission_id, address = %c.opoi_address,
                        "reveal not eligible/ready for this pool address, trying next"
                    );
                }
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
        let stake_pool = self.stake_pool.clone();
        let request_id = params.request_id.clone();
        let response_hash = params.response_hash.clone();
        let token_count = params.token_count;
        tokio::spawn(async move {
            do_commit(csd, pool, stake_pool, id, request_id, response_hash, token_count).await;
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
///
/// Commits with EVERY address in the stake pool concurrently, not just one:
/// REVEAL is VRF-gated (~3% eligibility per address, see stake_pool.rs) and
/// must use the same address that committed, so which address will turn
/// out eligible can't be known ahead of commit time — the only way to find
/// out is to have already committed with it. Each attempt's outcome is
/// persisted independently (`opoi_commit_attempts`); the submission overall
/// moves to COMMITTED as soon as at least one attempt succeeds, or FAILED
/// only if every pool address failed to commit.
///
/// `pub(crate)` (not private): also called by `shard_engine.rs` once a shard
/// pipeline's final token/shard result comes back, so a shard-routed
/// request's response drives the exact same commit -> reveal -> publish ->
/// payout lifecycle a whole-model response does, instead of a second
/// reimplementation. `poll_reveal_tick`/`do_publish` (this file) and
/// `payout::payout_tick` are all already DB-row-driven, not
/// `OpoiEngine`-instance-driven, so nothing else needed changing for a
/// shard-created `opoi_submissions` row to self-heal through the rest of
/// the lifecycle the same way — this function is the one place that
/// genuinely needed sharing.
pub(crate) async fn do_commit(
    csd: Arc<CsdRpcClient>,
    pool: PgPool,
    stake_pool: Arc<StakePool>,
    submission_id: i64,
    request_id: String,
    response_hash: String,
    token_count: u32,
) {
    let results: Vec<(String, Result<crate::rpc::types::CommitResult, AppError>)> = stream::iter(stake_pool.addresses().to_vec())
        .map(|addr| {
            let csd = csd.clone();
            let request_id = request_id.clone();
            let response_hash = response_hash.clone();
            async move {
                let res = csd.submit_response_commit(&request_id, &response_hash, &addr, token_count).await;
                (addr, res)
            }
        })
        .buffer_unordered(4)
        .collect()
        .await;

    let mut first_success: Option<(String, crate::rpc::types::CommitResult)> = None;

    for (addr, res) in results {
        match res {
            Ok(r) => {
                if let Err(e) =
                    db::repo::upsert_commit_attempt_success(&pool, submission_id, &addr, &r.txid, &r.nonce_hex, r.commit_window_closes_at_height as i32).await
                {
                    tracing::error!(error = %e, submission_id, address = %addr, "failed to persist COMMITTED attempt");
                } else {
                    tracing::info!(submission_id, request_id = %request_id, address = %addr, txid = %r.txid, "OPoI response committed on-chain (one pool address)");
                }
                if first_success.is_none() {
                    first_success = Some((addr, r));
                }
            }
            Err(e) => {
                let _ = db::repo::upsert_commit_attempt_failure(&pool, submission_id, &addr, &e.to_string()).await;
                tracing::debug!(error = %e, submission_id, address = %addr, "commit attempt failed for this pool address");
            }
        }
    }

    match first_success {
        Some((_addr, r)) => {
            if let Err(e) = db::repo::mark_committed(&pool, submission_id, &r.txid, &r.nonce_hex, r.commit_window_closes_at_height as i32).await {
                tracing::error!(error = %e, submission_id, "failed to persist overall COMMITTED state");
            }
        }
        None => {
            tracing::warn!(submission_id, request_id = %request_id, "every pool address failed to commit; marking FAILED");
            if let Err(e2) = db::repo::mark_failed(&pool, submission_id, "all stake-pool addresses failed to commit").await {
                tracing::error!(error = %e2, submission_id, "failed to persist FAILED state");
            }
        }
    }
}
