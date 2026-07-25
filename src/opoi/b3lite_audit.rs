//! B3-lite sampling queue + Auditor dispatch + consequence policy — see
//! `b3lite.rs`'s module doc for the receipt/signing half this builds on,
//! and "ESCOPO CONCRETO DA B3-LITE" in `CS COIN OPoI MELHOR IMPLEMENTAÇÃO.txt`
//! for the full design history.
//!
//! Dispatch design: this bridge runs the audit by invoking a LOCAL
//! `cs-miner` binary as a subprocess (`cs-miner --audit-request <file>`,
//! see that crate's `src/opoi/auditor.rs`/`cli.rs`), not by pushing an
//! audit assignment to some arbitrary CONNECTED miner over the stratum
//! wire protocol. This is a deliberate choice, not a shortcut: B3-lite's
//! off-chain consequence (withhold pay / reputation flag / eject) is
//! imposed unilaterally by THIS bridge operator, so the audit needs to be
//! something the OPERATOR trusts independently — asking a random pool
//! participant to audit a peer gives no extra security margin (that peer
//! has exactly as much self-interest as the one being audited, with no
//! stake/slashing backing its own honesty here — that's B3-full's
//! still-unbuilt on-chain fraud-oracle territory, not this). A trusted
//! local subprocess is both simpler AND the more correct trust model for
//! an off-chain-only mechanism.
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

use sqlx::PgPool;

use crate::db;
use crate::db::models::B3LiteReceipt;
use crate::miner_registry::MinerRegistry;
use crate::opoi::shard_engine::ModelSourceConfig;

/// Matches cs-miner's `opoi::auditor::AuditVerdict`/`PositionVerdict` JSON
/// shape (see that crate's `auditor.rs` — both derive `serde::Serialize`
/// with no rename attributes, so field names line up as-is). Duplicated
/// here rather than shared via a crate dependency: these are two separate
/// binaries/repos, and the wire format is JSON on stdout, not a Rust type.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct AuditVerdictJson {
    positions: Vec<AuditPositionJson>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct AuditPositionJson {
    #[allow(dead_code)]
    step: usize,
    #[allow(dead_code)]
    committed_token_id: u32,
    #[allow(dead_code)]
    auditor_token_id: u32,
    admissible: bool,
}

impl AuditVerdictJson {
    fn all_admissible(&self) -> bool {
        !self.positions.is_empty() && self.positions.iter().all(|p| p.admissible)
    }
}

pub struct B3LiteAuditor {
    db: PgPool,
    model_sources: ModelSourceConfig,
    registry: Arc<MinerRegistry>,
    cs_miner_bin: String,
    cache_dir: PathBuf,
}

/// A wallet accumulating this many confirmed-divergent audits, ever, is
/// additionally ejected from future OPoI dispatch — see module doc.
const EJECT_AFTER_DIVERGENCES: i64 = 3;

impl B3LiteAuditor {
    pub fn new(db: PgPool, model_sources: ModelSourceConfig, registry: Arc<MinerRegistry>, cs_miner_bin: String, cache_dir: PathBuf) -> Self {
        Self { db, model_sources, registry, cs_miner_bin, cache_dir }
    }

    /// One tick: replay every receipt sampled for audit that hasn't been
    /// audited yet. Sequential (not concurrent) deliberately — each replay
    /// spawns a whole model-loading subprocess, and running several at
    /// once would multiply peak RAM/CPU usage on whatever host runs this
    /// bridge; B3-lite's sample rate is meant to be small precisely so a
    /// sequential tick keeps up.
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

        let Some(source) = self.model_sources.get(&receipt.model_id).cloned() else {
            tracing::warn!(request_id = %request_id, model_id = %receipt.model_id, "b3lite audit: no MODEL_SOURCES_JSON entry, cannot resolve GGUF — marking INCONCLUSIVE");
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

        if let Err(e) = tokio::fs::create_dir_all(&self.cache_dir).await {
            tracing::warn!(error = %e, request_id = %request_id, "b3lite audit: could not create cache_dir — marking INCONCLUSIVE");
            self.mark_inconclusive(receipt.id, &format!("could not create cache_dir: {e}")).await;
            return;
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
            tracing::warn!(error = %e, request_id = %request_id, "b3lite audit: could not write request file — marking INCONCLUSIVE");
            self.mark_inconclusive(receipt.id, &format!("could not write request file: {e}")).await;
            return;
        }

        let output = tokio::process::Command::new(&self.cs_miner_bin)
            .arg("--address").arg("b3lite-auditor")
            .arg("--worker").arg("b3lite-auditor")
            .arg("--pool").arg("127.0.0.1:1")
            .arg("--audit-request").arg(&tmp_path)
            .output()
            .await;

        let _ = tokio::fs::remove_file(&tmp_path).await;

        let out = match output {
            Ok(out) => out,
            Err(e) => {
                tracing::warn!(error = %e, request_id = %request_id, bin = %self.cs_miner_bin, "b3lite audit: could not spawn auditor subprocess — marking INCONCLUSIVE");
                self.mark_inconclusive(receipt.id, &format!("could not spawn auditor subprocess: {e}")).await;
                return;
            }
        };

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            tracing::warn!(request_id = %request_id, status = ?out.status, stderr = %stderr, "b3lite audit: auditor subprocess exited non-zero — marking INCONCLUSIVE");
            self.mark_inconclusive(receipt.id, &format!("auditor subprocess exited {:?}: {stderr}", out.status)).await;
            return;
        }

        let verdict: AuditVerdictJson = match serde_json::from_slice(&out.stdout) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, request_id = %request_id, "b3lite audit: could not parse auditor stdout — marking INCONCLUSIVE");
                self.mark_inconclusive(receipt.id, &format!("could not parse auditor stdout: {e}")).await;
                return;
            }
        };

        let all_admissible = verdict.all_admissible();
        let status = if all_admissible { "ADMISSIBLE" } else { "DIVERGENT" };
        let detail = serde_json::to_string(&verdict).unwrap_or_default();

        if let Err(e) = db::repo::mark_b3lite_audit_result(&self.db, receipt.id, status, &detail).await {
            tracing::warn!(error = %e, request_id = %request_id, "b3lite audit: failed to persist audit result");
        }

        if all_admissible {
            tracing::info!(request_id = %request_id, receipt_id = receipt.id, "B3-lite audit: ADMISSIBLE");
        } else {
            tracing::warn!(request_id = %request_id, receipt_id = receipt.id, wallet = %receipt.miner_wallet, "B3-lite audit: DIVERGENT — applying off-chain consequence");
            self.apply_consequence(&receipt).await;
        }
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
    //! End-to-end smoke test for `audit_tick` itself — the one piece none
    //! of the other tests (pure-function unit tests here, the live-DB repo
    //! test in `db/repo.rs`, cs-miner's own CLI acceptance test) actually
    //! exercises together: a real receipt row, a real `ModelSourceConfig`,
    //! a real `cs-miner --audit-request` subprocess invocation, and real
    //! JSON parsed back out of its stdout. Gated on ALL of
    //! `B3LITE_TEST_DATABASE_URL` / `B3LITE_TEST_CS_MINER_BIN` /
    //! `B3LITE_TEST_MODEL_SOURCES_JSON` being set — skips otherwise, same
    //! reasoning as `db/repo.rs`'s live-DB test.
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
        let auditor = B3LiteAuditor::new(pool.clone(), model_sources, registry, cs_miner_bin, cache_dir.clone());

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

    #[test]
    fn audit_verdict_all_admissible_true_when_every_position_is() {
        let v = AuditVerdictJson {
            positions: vec![
                AuditPositionJson { step: 0, committed_token_id: 1, auditor_token_id: 1, admissible: true },
                AuditPositionJson { step: 1, committed_token_id: 2, auditor_token_id: 2, admissible: true },
            ],
        };
        assert!(v.all_admissible());
    }

    #[test]
    fn audit_verdict_all_admissible_false_on_any_divergent_position() {
        let v = AuditVerdictJson {
            positions: vec![
                AuditPositionJson { step: 0, committed_token_id: 1, auditor_token_id: 1, admissible: true },
                AuditPositionJson { step: 1, committed_token_id: 99, auditor_token_id: 2, admissible: false },
            ],
        };
        assert!(!v.all_admissible());
    }

    #[test]
    fn audit_verdict_all_admissible_false_when_empty() {
        let v = AuditVerdictJson { positions: vec![] };
        assert!(!v.all_admissible());
    }
}
