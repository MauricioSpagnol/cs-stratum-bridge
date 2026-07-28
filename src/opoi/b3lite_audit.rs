//! B3-lite sampling queue + Auditor dispatch + consequence policy — see
//! `b3lite.rs`'s module doc for the receipt/signing half this builds on,
//! and "ESCOPO CONCRETO DA B3-LITE" in `CS COIN OPoI MELHOR IMPLEMENTAÇÃO.txt`
//! for the full design history.
//!
//! Dispatch design: an audit only ever runs somewhere THIS bridge operator
//! trusts — never an arbitrary connected miner. Two paths, tried in order:
//!
//! 1. **Stratum, to an operator-designated wallet** (`opoi.audit_assign`/
//!    `opoi.audit_result`, see `pow::stratum.rs` on cs-miner's side): only
//!    ever dispatched to a wallet BOTH currently connected AND announced
//!    the `auditor` capability AND present in this operator's own
//!    `Config::auditor_trusted_wallets` allow-list (see
//!    `MinerRegistry::pick_auditor_capable`) — lets the operator run a
//!    dedicated auditor machine (or several) as a normal pool connection
//!    instead of needing local compute on the bridge host itself.
//! 2. **Local subprocess fallback** (`cs-miner --audit-request <file>`):
//!    always available, no dependency on any miner being connected.
//!
//! Deliberately NOT a third option — dispatching to an arbitrary connected
//! miner (not specifically designated trusted) — because that miner has
//! exactly as much self-interest as the one being audited, with no
//! stake/slashing backing its own honesty here (that's B3-full's
//! still-unbuilt on-chain fraud-oracle territory, not this).
//!
//! Consequence policy (see `apply_consequence`): every confirmed DIVERGENT
//! audit withholds pay for that one request and flags the wallet's
//! reputation; a wallet accumulating 3 or more confirmed divergences ever
//! is additionally ejected from future OPoI dispatch (PoW mining is
//! unaffected — see `miner_registry::MinerRegistry::ban`'s doc). The
//! threshold is a simple, defensible starting point, not empirically
//! tuned — B3-lite's own explicit purpose (per the scope doc) is to
//! GENERATE the real-world data that calibrates this kind of threshold
//! before anything like it goes on-chain (B3-full).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use sqlx::PgPool;
use tokio::sync::oneshot;

use crate::db;
use crate::db::models::B3LiteReceipt;
use crate::error::AppError;
use crate::miner_registry::MinerRegistry;
use crate::opoi::shard_engine::{ModelSource, ModelSourceConfig};
use crate::opoi::wire::{self, AuditPositionWire, OpoiAuditResult};

/// Subprocess-stdout counterpart of `wire::OpoiAuditResult` — same
/// `positions` shape, but cs-miner's `--audit-request` CLI prints only
/// `{"positions":[...]}` (no `request_id` wrapper) to stdout, unlike the
/// stratum wire message.
#[derive(Debug, serde::Deserialize)]
struct AuditStdoutJson {
    positions: Vec<AuditPositionWire>,
}

fn all_admissible(positions: &[AuditPositionWire]) -> bool {
    !positions.is_empty() && positions.iter().all(|p| p.admissible)
}

/// How long to wait for a dispatched-to-stratum auditor to answer before
/// falling back to the local subprocess — generous: a real forward pass
/// over a multi-GB GGUF (possibly after downloading it fresh) can
/// legitimately take a couple of minutes on modest hardware.
const STRATUM_AUDIT_TIMEOUT: Duration = Duration::from_secs(180);

/// A wallet accumulating this many confirmed-divergent audits, ever, is
/// additionally ejected from future OPoI dispatch — see module doc.
const EJECT_AFTER_DIVERGENCES: i64 = 3;

pub struct B3LiteAuditor {
    db: PgPool,
    model_sources: ModelSourceConfig,
    registry: Arc<MinerRegistry>,
    cs_miner_bin: String,
    cache_dir: PathBuf,
    trusted_wallets: Vec<String>,
    /// Outstanding stratum audit dispatches: `request_id` -> the wallet
    /// expected to answer (assignment-verification, same contract
    /// `ShardEngine::assignments` gives shard dispatch) — checked by
    /// `handle_audit_result` before accepting a reply.
    audit_assignments: DashMap<String, String>,
    /// Outstanding stratum audit dispatches: `request_id` -> a one-shot
    /// sender `dispatch_via_stratum` is awaiting on. Removed (and the
    /// oneshot fires) the moment a matching `opoi.audit_result` arrives.
    pending: DashMap<String, oneshot::Sender<Vec<AuditPositionWire>>>,
}

impl B3LiteAuditor {
    pub fn new(
        db: PgPool,
        model_sources: ModelSourceConfig,
        registry: Arc<MinerRegistry>,
        cs_miner_bin: String,
        cache_dir: PathBuf,
        trusted_wallets: Vec<String>,
    ) -> Self {
        Self { db, model_sources, registry, cs_miner_bin, cache_dir, trusted_wallets, audit_assignments: DashMap::new(), pending: DashMap::new() }
    }

    /// One tick: replay every receipt sampled for audit that hasn't been
    /// audited yet. Sequential (not concurrent) deliberately — even the
    /// stratum path ties up ONE trusted auditor connection at a time, and
    /// the subprocess fallback spawns a whole model-loading process;
    /// running several at once would multiply peak RAM/CPU usage on
    /// whatever host(s) run this. B3-lite's sample rate is meant to be
    /// small precisely so a sequential tick keeps up.
    pub async fn audit_tick(&self) {
        let pending = match db::repo::list_pending_b3lite_audits(&self.db).await {
            Ok(rows) => rows,
            Err(e) => {
                tracing::warn!(error = %e, "b3lite audit_tick: failed to list pending audits");
                return;
            }
        };

        for receipt in pending {
            self.run_one(receipt).await;
        }
    }

    async fn run_one(&self, receipt: B3LiteReceipt) {
        let request_id = receipt.request_id.clone();

        let Some(source) = self.model_sources.resolve(&receipt.model_id).await else {
            tracing::warn!(request_id = %request_id, model_id = %receipt.model_id, "b3lite audit: no model source (local override or cs-marketplace), cannot resolve GGUF — marking INCONCLUSIVE");
            self.mark_inconclusive(receipt.id, "no ModelSource entry for this model_id").await;
            return;
        };
        let Some(prompt_text) = decode_prompt_hex(&receipt.prompt_hex) else {
            tracing::warn!(request_id = %request_id, "b3lite audit: stored prompt_hex is not valid UTF-8 hex — marking INCONCLUSIVE");
            self.mark_inconclusive(receipt.id, "prompt_hex did not decode to valid UTF-8").await;
            return;
        };
        let Some(committed_token_ids) = decode_token_ids_hex(&receipt.generated_token_ids_hex) else {
            tracing::warn!(request_id = %request_id, "b3lite audit: stored generated_token_ids_hex is malformed — marking INCONCLUSIVE");
            self.mark_inconclusive(receipt.id, "generated_token_ids_hex was not a multiple of 4 bytes").await;
            return;
        };

        let verdict_result = match self.registry.pick_auditor_capable(&self.trusted_wallets, &receipt.miner_wallet) {
            Some(wallet) => {
                tracing::info!(request_id = %request_id, wallet = %wallet, "b3lite audit: dispatching via stratum to a trusted auditor");
                match self.dispatch_via_stratum(&wallet, &receipt, &source, &prompt_text, &committed_token_ids).await {
                    Ok(positions) => Ok(positions),
                    Err(e) => {
                        tracing::warn!(request_id = %request_id, wallet = %wallet, error = %e, "b3lite audit: stratum dispatch failed, falling back to local subprocess");
                        self.dispatch_via_subprocess(&receipt, &source, &prompt_text, &committed_token_ids).await
                    }
                }
            }
            None => self.dispatch_via_subprocess(&receipt, &source, &prompt_text, &committed_token_ids).await,
        };

        let positions = match verdict_result {
            Ok(p) => p,
            Err(e) => {
                self.mark_inconclusive(receipt.id, &e).await;
                return;
            }
        };

        let admissible = all_admissible(&positions);
        let status = if admissible { "ADMISSIBLE" } else { "DIVERGENT" };
        let detail = serde_json::to_string(&positions).unwrap_or_default();

        if let Err(e) = db::repo::mark_b3lite_audit_result(&self.db, receipt.id, status, &detail).await {
            tracing::warn!(error = %e, request_id = %request_id, "b3lite audit: failed to persist audit result");
        }

        if admissible {
            tracing::info!(request_id = %request_id, receipt_id = receipt.id, "B3-lite audit: ADMISSIBLE");
        } else {
            tracing::warn!(request_id = %request_id, receipt_id = receipt.id, wallet = %receipt.miner_wallet, "B3-lite audit: DIVERGENT — applying off-chain consequence");
            self.apply_consequence(&receipt).await;
        }
    }

    /// Dispatches to `wallet` over stratum and awaits its `opoi.audit_result`
    /// (or a timeout). `Err` covers every reason this path didn't produce a
    /// verdict — the caller falls back to the local subprocess on any of
    /// them, so the exact wording is diagnostic only.
    async fn dispatch_via_stratum(
        &self,
        wallet: &str,
        receipt: &B3LiteReceipt,
        source: &ModelSource,
        prompt_text: &str,
        committed_token_ids: &[u32],
    ) -> Result<Vec<AuditPositionWire>, String> {
        let tx = self.registry.get(wallet).ok_or_else(|| format!("wallet {wallet} not connected"))?;

        let assign = wire::OpoiAuditAssign {
            request_id: receipt.request_id.clone(),
            model_id: receipt.model_id.clone(),
            gguf_url: source.gguf_url.clone(),
            gguf_sha256: receipt.gguf_sha256.clone(),
            tokenizer_url: source.tokenizer_url.clone(),
            tokenizer_sha256: source.tokenizer_sha256.clone(),
            prompt_hex: hex::encode(prompt_text.as_bytes()),
            committed_token_ids: committed_token_ids.to_vec(),
            total_layers: receipt.total_layers as u32,
        };

        let (result_tx, result_rx) = oneshot::channel();
        self.audit_assignments.insert(receipt.request_id.clone(), wallet.to_string());
        self.pending.insert(receipt.request_id.clone(), result_tx);

        if tx.send(wire::build_audit_assign_line(&assign)).is_err() {
            self.audit_assignments.remove(&receipt.request_id);
            self.pending.remove(&receipt.request_id);
            return Err(format!("wallet {wallet}'s downstream channel is closed"));
        }

        let outcome = tokio::time::timeout(STRATUM_AUDIT_TIMEOUT, result_rx).await;

        // Always clean up both maps regardless of outcome — a late-arriving
        // `opoi.audit_result` after a timeout must not be silently accepted
        // for whatever NEXT dispatch happens to reuse this request_id (it
        // won't in practice, request_ids are unique per on-chain request,
        // but this is the same "never trust a stale entry" discipline
        // `ShardEngine::assignments` already follows).
        self.audit_assignments.remove(&receipt.request_id);
        self.pending.remove(&receipt.request_id);

        match outcome {
            Ok(Ok(positions)) => Ok(positions),
            Ok(Err(_)) => Err(format!("audit oneshot dropped for wallet {wallet}")),
            Err(_) => Err(format!("audit via wallet {wallet} timed out after {STRATUM_AUDIT_TIMEOUT:?}")),
        }
    }

    async fn dispatch_via_subprocess(
        &self,
        receipt: &B3LiteReceipt,
        source: &ModelSource,
        prompt_text: &str,
        committed_token_ids: &[u32],
    ) -> Result<Vec<AuditPositionWire>, String> {
        if let Err(e) = tokio::fs::create_dir_all(&self.cache_dir).await {
            return Err(format!("could not create cache_dir: {e}"));
        }

        let request_json = serde_json::json!({
            "model_id": receipt.model_id,
            "gguf_url": source.gguf_url,
            "gguf_sha256": receipt.gguf_sha256,
            "tokenizer_url": source.tokenizer_url,
            "tokenizer_sha256": source.tokenizer_sha256,
            "cache_dir": self.cache_dir,
            "prompt_text": prompt_text,
            "committed_token_ids": committed_token_ids,
            "total_layers": receipt.total_layers,
        });

        let tmp_path = self.cache_dir.join(format!("audit_req_{}.json", receipt.id));
        let write_result = match serde_json::to_vec(&request_json) {
            Ok(bytes) => tokio::fs::write(&tmp_path, bytes).await,
            Err(e) => Err(std::io::Error::other(e)),
        };
        if let Err(e) = write_result {
            return Err(format!("could not write request file: {e}"));
        }

        let output = tokio::process::Command::new(&self.cs_miner_bin)
            .arg("--address").arg("b3lite-auditor")
            .arg("--worker").arg("b3lite-auditor")
            .arg("--pool").arg("127.0.0.1:1")
            .arg("--audit-request").arg(&tmp_path)
            .output()
            .await;

        let _ = tokio::fs::remove_file(&tmp_path).await;

        let out = output.map_err(|e| format!("could not spawn auditor subprocess ({}): {e}", self.cs_miner_bin))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!("auditor subprocess exited {:?}: {stderr}", out.status));
        }

        let parsed: AuditStdoutJson =
            serde_json::from_slice(&out.stdout).map_err(|e| format!("could not parse auditor stdout: {e}"))?;
        Ok(parsed.positions)
    }

    async fn mark_inconclusive(&self, receipt_id: i64, detail: &str) {
        if let Err(e) = db::repo::mark_b3lite_audit_result(&self.db, receipt_id, "INCONCLUSIVE", detail).await {
            tracing::warn!(error = %e, receipt_id, "b3lite audit: failed to persist INCONCLUSIVE result");
        }
    }

    /// See module doc's "Consequence policy". `receipt.id`/`request_id`
    /// tie every consequence row back to the specific audit that produced
    /// it, even the `EJECTED` one, which is really about the wallet's
    /// cumulative history rather than this one request.
    async fn apply_consequence(&self, receipt: &B3LiteReceipt) {
        let reason = format!("B3-lite audit found a real divergence for request_id={}", receipt.request_id);

        if let Err(e) = db::repo::insert_b3lite_consequence(&self.db, receipt.id, &receipt.miner_wallet, &receipt.request_id, "WITHHOLD_PAY", &reason).await {
            tracing::warn!(error = %e, receipt_id = receipt.id, "failed to record WITHHOLD_PAY consequence");
        }
        if let Err(e) = db::repo::insert_b3lite_consequence(&self.db, receipt.id, &receipt.miner_wallet, &receipt.request_id, "REPUTATION_FLAG", &reason).await {
            tracing::warn!(error = %e, receipt_id = receipt.id, "failed to record REPUTATION_FLAG consequence");
        }

        match db::repo::count_withhold_consequences(&self.db, &receipt.miner_wallet).await {
            Ok(n) if n >= EJECT_AFTER_DIVERGENCES => {
                self.registry.ban(&receipt.miner_wallet);
                let eject_reason = format!("{n} confirmed B3-lite divergences (threshold {EJECT_AFTER_DIVERGENCES})");
                if let Err(e) = db::repo::insert_b3lite_consequence(&self.db, receipt.id, &receipt.miner_wallet, &receipt.request_id, "EJECTED", &eject_reason).await {
                    tracing::warn!(error = %e, receipt_id = receipt.id, "failed to record EJECTED consequence");
                }
                tracing::warn!(wallet = %receipt.miner_wallet, count = n, "B3-lite: wallet ejected from future OPoI dispatch after repeated confirmed divergence");
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, wallet = %receipt.miner_wallet, "failed to count prior WITHHOLD_PAY consequences"),
        }
    }
}

#[async_trait::async_trait]
impl crate::opoi::handler::AuditHandler for B3LiteAuditor {
    async fn handle_audit_result(&self, wallet: &str, result: OpoiAuditResult) -> Result<(), AppError> {
        {
            let Some(assignment) = self.audit_assignments.get(&result.request_id) else {
                return Err(AppError::UnknownRequest);
            };
            if assignment.value() != wallet {
                return Err(AppError::NotAssignedToCaller);
            }
        }

        let Some((_, sender)) = self.pending.remove(&result.request_id) else {
            // Already timed out (dispatch_via_stratum's own cleanup already
            // removed both entries) — a late reply, not an error to the
            // caller, but nothing to deliver it to either.
            return Ok(());
        };
        let _ = sender.send(result.positions);
        Ok(())
    }

    async fn on_disconnect(&self, wallet: &str) {
        // Drop (fail fast, not "wait for timeout") any outstanding
        // dispatch whose expected wallet just disconnected — the oneshot's
        // Sender dropping resolves dispatch_via_stratum's `.await` as
        // Err immediately instead of idling until STRATUM_AUDIT_TIMEOUT.
        let stale: Vec<String> = self.audit_assignments.iter().filter(|e| e.value() == wallet).map(|e| e.key().clone()).collect();
        for request_id in stale {
            self.audit_assignments.remove(&request_id);
            self.pending.remove(&request_id); // dropping the Sender is enough to unblock the waiter
        }
    }
}

/// Reverses the hex-encoding `http/handlers.rs::submit_prompt` applies at
/// intake (`hex::encode(body.prompt.as_bytes())`) — the same round trip
/// every miner's own `ShardInputWire::Prompt` handling already performs.
fn decode_prompt_hex(hex_str: &str) -> Option<String> {
    let bytes = hex::decode(hex_str).ok()?;
    String::from_utf8(bytes).ok()
}

/// Reverses `shard_engine::build_response`'s encoding (each token id as 4
/// bytes little-endian, concatenated in generation order).
fn decode_token_ids_hex(hex_str: &str) -> Option<Vec<u32>> {
    let bytes = hex::decode(hex_str).ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[cfg(test)]
mod live_audit_tick_test {
    //! End-to-end smoke test for `audit_tick` itself (subprocess path) —
    //! see `db/repo.rs`'s live-DB test for the same gating reasoning.
    use super::*;
    use crate::miner_registry::MinerRegistry;

    #[tokio::test]
    async fn audit_tick_runs_a_real_receipt_through_a_real_cs_miner_subprocess() {
        let (Ok(db_url), Ok(cs_miner_bin), Ok(model_sources_json)) = (
            std::env::var("B3LITE_TEST_DATABASE_URL"),
            std::env::var("B3LITE_TEST_CS_MINER_BIN"),
            std::env::var("B3LITE_TEST_MODEL_SOURCES_JSON"),
        ) else {
            eprintln!("B3LITE_TEST_DATABASE_URL/B3LITE_TEST_CS_MINER_BIN/B3LITE_TEST_MODEL_SOURCES_JSON not all set — skipping live audit_tick smoke test");
            return;
        };

        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&db_url).await.expect("connect to test DB");
        sqlx::migrate!("./migrations").run(&pool).await.expect("run migrations");

        std::env::set_var("MODEL_SOURCES_JSON", &model_sources_json);
        let model_sources = ModelSourceConfig::from_env();

        // Real, honestly-generated receipt: same prompt/committed-token
        // fixture cs-miner's own `examples/auditor_test.rs` proved passes.
        let prompt_hex = hex::encode("The capital of France is".as_bytes());
        let generated_token_ids_hex = {
            let ids: [u32; 5] = [12095, 13, 1084, 374, 279];
            let mut raw = Vec::new();
            for id in ids {
                raw.extend_from_slice(&id.to_le_bytes());
            }
            hex::encode(raw)
        };

        let request_id = format!("audit-tick-test-{}", uuid::Uuid::new_v4());
        // Real sha256 of the QWEN2_5_0_5B.gguf fixture cs-miner's own tests
        // already use (target/model_fetch_test/peer_cache/) — must match
        // what B3LITE_TEST_MODEL_SOURCES_JSON's gguf_url actually serves,
        // or the auditor subprocess's gguf_fetch hash check will reject it.
        let gguf_sha256 = "c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";
        let receipt_id = db::repo::create_b3lite_receipt(
            &pool, &request_id, "test-wallet", "QWEN2_5_0_5B", gguf_sha256,
            None, &prompt_hex, "resp-hash-placeholder", &generated_token_ids_hex, 24, "sig", true,
        ).await.expect("create receipt");

        let cache_dir = std::env::temp_dir().join(format!("b3lite_audit_test_{receipt_id}"));
        let registry = std::sync::Arc::new(MinerRegistry::new());
        // No trusted wallets configured — this test exercises the
        // subprocess fallback path only (dedicated stratum-path tests live
        // separately, see `stratum_dispatch_tests` below).
        let auditor = B3LiteAuditor::new(pool.clone(), model_sources, registry, cs_miner_bin, cache_dir.clone(), vec![]);

        auditor.audit_tick().await;

        let row: (String,) = sqlx::query_as("SELECT audit_status FROM b3lite_receipts WHERE id = $1")
            .bind(receipt_id)
            .fetch_one(&pool)
            .await
            .expect("fetch audited receipt");
        assert_eq!(row.0, "ADMISSIBLE", "an honest, real receipt should audit as ADMISSIBLE");

        let _ = std::fs::remove_dir_all(&cache_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_prompt_hex_round_trips() {
        let original = "The capital of France is";
        let encoded = hex::encode(original.as_bytes());
        assert_eq!(decode_prompt_hex(&encoded), Some(original.to_string()));
    }

    #[test]
    fn decode_prompt_hex_rejects_invalid_hex() {
        assert_eq!(decode_prompt_hex("not-hex!!"), None);
    }

    #[test]
    fn decode_token_ids_hex_round_trips() {
        let ids = vec![1u32, 256, 65536, u32::MAX];
        let mut raw = Vec::new();
        for id in &ids {
            raw.extend_from_slice(&id.to_le_bytes());
        }
        let encoded = hex::encode(&raw);
        assert_eq!(decode_token_ids_hex(&encoded), Some(ids));
    }

    #[test]
    fn decode_token_ids_hex_rejects_non_multiple_of_4() {
        assert_eq!(decode_token_ids_hex("aabbcc"), None); // 3 bytes
    }

    #[test]
    fn decode_token_ids_hex_empty_is_empty_vec() {
        assert_eq!(decode_token_ids_hex(""), Some(vec![]));
    }

    fn position(admissible: bool) -> AuditPositionWire {
        AuditPositionWire { step: 0, committed_token_id: 1, auditor_token_id: 1, admissible }
    }

    #[test]
    fn all_admissible_true_when_every_position_is() {
        assert!(all_admissible(&[position(true), position(true)]));
    }

    #[test]
    fn all_admissible_false_on_any_divergent_position() {
        assert!(!all_admissible(&[position(true), position(false)]));
    }

    #[test]
    fn all_admissible_false_when_empty() {
        assert!(!all_admissible(&[]));
    }
}

#[cfg(test)]
mod stratum_dispatch_tests {
    //! Unit tests for `dispatch_via_stratum`/`handle_audit_result`'s
    //! correlation logic — no live DB/subprocess needed, since these never
    //! reach `run_one`'s DB-writing tail. Uses a fake downstream channel
    //! (an `UnboundedReceiver` this test reads from directly) instead of a
    //! real stratum socket, same boundary `MinerRegistry` already draws
    //! (it only ever sees a channel, never a socket).
    use super::*;
    use crate::opoi::handler::AuditHandler;
    use crate::opoi::wire::{AuditPositionWire, OpoiAuditResult};

    fn auditor_with_registry() -> (B3LiteAuditor, Arc<MinerRegistry>) {
        let registry = Arc::new(MinerRegistry::new());
        // A real PgPool is never touched by these tests (they never reach
        // `run_one`'s DB-writing tail) — `connect_lazy` builds a pool
        // object without dialing anything.
        let db = PgPool::connect_lazy("postgres://unused/unused").expect("lazy pool");
        let auditor = B3LiteAuditor::new(db, ModelSourceConfig::from_env(), registry.clone(), "unused".into(), PathBuf::from("/tmp"), vec!["trusted-wallet".into()]);
        (auditor, registry)
    }

    #[tokio::test]
    async fn dispatch_via_stratum_delivers_verdict_on_matching_reply() {
        let (auditor, registry) = auditor_with_registry();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("trusted-wallet".to_string(), tx);
        registry.mark_auditor_capable("trusted-wallet");

        let source = ModelSource { gguf_url: "http://x/gguf".into(), tokenizer_url: "http://x/tok".into(), tokenizer_sha256: "tok-hash".into() };
        let receipt = B3LiteReceipt {
            id: 1, request_id: "req-1".into(), miner_wallet: "audited-wallet".into(), model_id: "M".into(),
            gguf_sha256: "gguf-hash".into(), prompt_hash: None, prompt_hex: hex::encode("hi"), response_hash: "r".into(),
            generated_token_ids_hex: String::new(), total_layers: 4, signature_hex: "sig".into(), sampled: true,
            audit_status: "PENDING".into(), audit_detail: None, created_at: chrono::Utc::now(), audited_at: None,
        };

        let dispatch = tokio::spawn(async move { auditor.dispatch_via_stratum("trusted-wallet", &receipt, &source, "hi", &[1, 2]).await });

        // Drain the pushed opoi.audit_assign line (proves it was actually sent).
        let line = rx.recv().await.expect("assign line sent");
        assert!(line.contains("opoi.audit_assign"));
        assert!(line.contains("\"request_id\":\"req-1\""));

        // Note: can't call auditor.handle_audit_result here — auditor was
        // moved into the spawned task. This test only proves the assign
        // line is sent correctly; the correlation itself is proven by
        // `handle_audit_result_delivers_to_the_matching_pending_dispatch`
        // below using the same auditor instance directly (no spawn).
        dispatch.abort();
    }

    #[tokio::test]
    async fn handle_audit_result_rejects_wrong_wallet() {
        let (auditor, registry) = auditor_with_registry();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("trusted-wallet".to_string(), tx);
        auditor.audit_assignments.insert("req-1".to_string(), "trusted-wallet".to_string());
        auditor.pending.insert("req-1".to_string(), oneshot::channel().0);

        let result = OpoiAuditResult { request_id: "req-1".into(), positions: vec![] };
        let outcome = auditor.handle_audit_result("some-other-wallet", result).await;
        assert!(matches!(outcome, Err(AppError::NotAssignedToCaller)));
    }

    #[tokio::test]
    async fn handle_audit_result_rejects_unknown_request() {
        let (auditor, _registry) = auditor_with_registry();
        let result = OpoiAuditResult { request_id: "never-dispatched".into(), positions: vec![] };
        let outcome = auditor.handle_audit_result("any-wallet", result).await;
        assert!(matches!(outcome, Err(AppError::UnknownRequest)));
    }

    #[tokio::test]
    async fn handle_audit_result_delivers_to_the_matching_pending_dispatch() {
        let (auditor, registry) = auditor_with_registry();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("trusted-wallet".to_string(), tx);

        let (result_tx, result_rx) = oneshot::channel();
        auditor.audit_assignments.insert("req-1".to_string(), "trusted-wallet".to_string());
        auditor.pending.insert("req-1".to_string(), result_tx);

        let positions = vec![AuditPositionWire { step: 0, committed_token_id: 1, auditor_token_id: 1, admissible: true }];
        let result = OpoiAuditResult { request_id: "req-1".into(), positions: positions.clone() };
        auditor.handle_audit_result("trusted-wallet", result).await.expect("should be accepted");

        let delivered = result_rx.await.expect("oneshot should have fired");
        assert_eq!(delivered.len(), positions.len());
        assert_eq!(delivered[0].admissible, positions[0].admissible);
    }

    #[tokio::test]
    async fn on_disconnect_unblocks_a_waiting_dispatch_immediately() {
        let (auditor, registry) = auditor_with_registry();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("trusted-wallet".to_string(), tx);

        let (result_tx, result_rx) = oneshot::channel::<Vec<AuditPositionWire>>();
        auditor.audit_assignments.insert("req-1".to_string(), "trusted-wallet".to_string());
        auditor.pending.insert("req-1".to_string(), result_tx);

        auditor.on_disconnect("trusted-wallet").await;

        // The Sender was dropped (removed from `pending`, not fulfilled),
        // so awaiting the Receiver resolves to an error immediately
        // instead of hanging until STRATUM_AUDIT_TIMEOUT.
        assert!(result_rx.await.is_err());
        assert!(!auditor.audit_assignments.contains_key("req-1"));
    }
}
