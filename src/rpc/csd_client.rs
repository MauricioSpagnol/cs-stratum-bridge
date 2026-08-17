//! JSON-RPC 1.0 client for the `csd` daemon — plain HTTP POST + basic auth,
//! same envelope shape as the sibling `back-pool` Node.js client.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::de::DeserializeOwned;

use crate::error::AppError;
use crate::rpc::types::*;

/// Tolerant subset of `getinfo`'s result — extra daemon fields are ignored.
#[derive(Debug, Clone, serde::Deserialize)]
struct GetInfoResult {
    blocks: u64,
}

/// `{ txid }`-shaped results, reused by `renewopoistake` and `submitopoiresponse`.
#[derive(Debug, Clone, serde::Deserialize)]
struct TxidResult {
    txid: String,
}

pub struct CsdRpcClient {
    http: reqwest::Client,
    rpc_url: String,
    user: String,
    pass: String,
}

impl CsdRpcClient {
    pub fn new(rpc_url: String, user: String, pass: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
        Self {
            http,
            rpc_url,
            user,
            pass,
        }
    }

    /// POSTs a JSON-RPC 1.0 envelope and unwraps `result`, mapping any
    /// transport or RPC-level failure to `AppError::Rpc`.
    async fn call<T: DeserializeOwned>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<T, AppError> {
        let body = serde_json::json!({
            "jsonrpc": "1.0",
            "id": "cs-stratum-bridge",
            "method": method,
            "params": params,
        });

        let resp = self
            .http
            .post(&self.rpc_url)
            .basic_auth(&self.user, Some(&self.pass))
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::Rpc(format!("{method}: request failed: {e}")))?;

        let resp_body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AppError::Rpc(format!("{method}: failed to parse response body: {e}")))?;

        if let Some(err) = resp_body.get("error") {
            if !err.is_null() {
                return Err(AppError::Rpc(format!("{method}: {err}")));
            }
        }

        let result = resp_body.get("result").cloned().unwrap_or(serde_json::Value::Null);
        serde_json::from_value(result)
            .map_err(|e| AppError::Rpc(format!("{method}: failed to deserialize result: {e}")))
    }

    /// Startup/liveness check — also the source of truth for "daemon unreachable".
    pub async fn get_chain_height(&self) -> Result<u64, AppError> {
        let res: GetInfoResult = self.call("getinfo", serde_json::json!([])).await?;
        Ok(res.blocks)
    }

    /// Reads current stake status; used before commit/reveal to gate on ACTIVE.
    pub async fn get_opoi_stake(&self, miner_address: &str) -> Result<Option<OpoiStake>, AppError> {
        match self
            .call::<OpoiStake>("getopoistake", serde_json::json!([miner_address]))
            .await
        {
            Ok(stake) => Ok(Some(stake)),
            Err(AppError::Rpc(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Initial stake creation — OPoI lifecycle entry point.
    pub async fn stake_opoi(
        &self,
        miner_address: &str,
        collateral_txid: &str,
        collateral_vout: u32,
        endpoint: &str,
        model_id: &str,
    ) -> Result<StakeOpoiResult, AppError> {
        let params = serde_json::json!([
            miner_address,
            collateral_txid,
            collateral_vout,
            [],
            endpoint,
            model_id,
            0
        ]);
        self.call("stakeopoi", params).await
    }

    /// Periodic keep-alive to avoid stake expiry/suspension.
    pub async fn renew_opoi_stake(&self, miner_address: &str) -> Result<String, AppError> {
        let res: TxidResult = self
            .call("renewopoistake", serde_json::json!([miner_address]))
            .await?;
        Ok(res.txid)
    }

    /// Polls for inference work to hand out to downstream miners.
    pub async fn list_pending_requests(&self) -> Result<Vec<OpoiRequest>, AppError> {
        self.call("listopoirequests", serde_json::json!(["PENDING"]))
            .await
    }

    /// Fetches a single request, e.g. to re-check state before commit/reveal.
    pub async fn get_request(&self, request_id: &str) -> Result<OpoiRequest, AppError> {
        self.call("getopoirequest", serde_json::json!([request_id]))
            .await
    }

    /// Commit step of the commit-reveal OPoI response protocol.
    pub async fn submit_response_commit(
        &self,
        request_id: &str,
        response_hash_hex: &str,
        miner_address: &str,
        token_count: u32,
    ) -> Result<CommitResult, AppError> {
        let params = serde_json::json!([
            request_id,
            response_hash_hex,
            miner_address,
            "",
            token_count
        ]);
        self.call("submitopoiresponsecommit", params).await
    }

    /// Reveal step of the commit-reveal OPoI response protocol.
    pub async fn submit_response_reveal(
        &self,
        request_id: &str,
        response_hash_hex: &str,
        miner_address: &str,
        token_count: u32,
        nonce_hex: &str,
    ) -> Result<String, AppError> {
        let params = serde_json::json!([
            request_id,
            response_hash_hex,
            miner_address,
            "",
            token_count,
            nonce_hex
        ]);
        let res: TxidResult = self.call("submitopoiresponse", params).await?;
        Ok(res.txid)
    }

    /// Publishes raw response/content bytes for a request; no signature needed.
    pub async fn submit_content(
        &self,
        request_id: &str,
        kind: &str,
        content_hex: &str,
    ) -> Result<(), AppError> {
        let params = serde_json::json!([request_id, kind, content_hex]);
        let _: serde_json::Value = self.call("submitopoicontent", params).await?;
        Ok(())
    }

    /// F15-H (Sessão 3): fetches a model's manifest — arch_type/num_layers to
    /// size the shard pipeline, backbone_pom_root to verify the GGUF a miner
    /// downloads, status to gate on ACTIVE. Read-only, no miner_address.
    pub async fn get_model_manifest(&self, model_id: &str) -> Result<ModelManifest, AppError> {
        self.call("getmodelmanifest", serde_json::json!([model_id])).await
    }

    /// F15-H (Sessão 3): fetches the Model Execution Graph — the ordered
    /// list of shards `shard_engine.rs` dispatches in sequence. Read-only.
    pub async fn get_model_graph(&self, model_id: &str) -> Result<ModelGraph, AppError> {
        self.call("getmodelgraph", serde_json::json!([model_id])).await
    }

    /// F15-H (Sessão 3): VRF self-claim of the coordinator role for a
    /// request's shard pipeline. Fails with an RPC error ("VRF proof
    /// generation failed") if `miner_address` isn't eligible this round —
    /// callers should treat that as "try the next pool address," never as
    /// a fatal error (see `stake_pool.rs`).
    pub async fn claim_coordinator(&self, request_id: &str, miner_address: &str) -> Result<ClaimResult, AppError> {
        self.call("claimcoordinator", serde_json::json!([request_id, miner_address])).await
    }

    /// F15-H (Sessão 3): VRF self-claim + publish of one shard's real output
    /// hash. Same "not eligible this round" failure mode as
    /// `claim_coordinator` — never fatal, just means this pool address isn't
    /// the one VRF selected for this (request, shard_index) pair.
    pub async fn submit_shard_result(
        &self,
        request_id: &str,
        shard_index: u32,
        miner_address: &str,
        boundary_output_hash_hex: &str,
        token_count: u32,
    ) -> Result<ShardResultTxRes, AppError> {
        let params = serde_json::json!([request_id, shard_index, miner_address, boundary_output_hash_hex, "", token_count]);
        self.call("submitshardresult", params).await
    }

    /// Batches miner payouts in a single wallet transaction.
    ///
    /// `sendmany` is a standard Bitcoin-Core-family wallet RPC, inherited by
    /// this fork — not OPoI-specific and not verified against this project's
    /// actual RPC allowlist in this session; confirm it's enabled/registered
    /// before relying on it in production.
    ///
    /// Amounts are formatted as fixed 8-decimal-place strings rather than
    /// passed through as raw JSON numbers: summing rewards in `f64` easily
    /// produces values like `3.0191999999999997` (`1.0064 * 3`) that have no
    /// exact decimal representation, and serde_json's shortest-round-trip
    /// serializer emits every one of those digits. The daemon's
    /// `ParseFixedPoint(str, 8, ...)` rejects anything past 8 decimal places
    /// with `RPC_TYPE_ERROR`/"Invalid amount" — confirmed live against a real
    /// payout tick. `AmountFromValue` on the daemon side accepts string
    /// amounts too, so formatting here sidesteps the float round-trip
    /// entirely instead of trying to pick a "safe" rounding of the f64.
    /// Lists spendable UTXOs in the daemon's own wallet — used by the setup
    /// wizard (`setup_wizard.rs`) to let the operator pick one as OPoI stake
    /// collateral instead of having to look up a txid/vout by hand.
    pub async fn list_unspent(&self) -> Result<Vec<UnspentOutput>, AppError> {
        self.call("listunspent", serde_json::json!([])).await
    }

    pub async fn send_many(
        &self,
        from_account_or_empty: &str,
        amounts: &BTreeMap<String, f64>,
    ) -> Result<String, AppError> {
        let params = serde_json::json!([from_account_or_empty, format_amounts(amounts)]);
        self.call("sendmany", params).await
    }
}

/// Renders each amount as a fixed 8-decimal-place string — see `send_many`'s
/// doc comment for why raw `f64` JSON serialization isn't safe here.
fn format_amounts(amounts: &BTreeMap<String, f64>) -> BTreeMap<String, String> {
    amounts.iter().map(|(wallet, amount)| (wallet.clone(), format!("{:.8}", amount))).collect()
}

#[cfg(test)]
mod format_amounts_tests {
    use super::*;

    #[test]
    fn renders_lossy_float_sums_within_the_daemons_8_decimal_limit() {
        let mut amounts = BTreeMap::new();
        // 1.0064 * 3 in real f64 arithmetic — the exact value that triggered
        // a live `sendmany` "Invalid amount" failure before this fix.
        amounts.insert("tmAVMxFD4vAvu16Pefa2X66zixCWrYFpcUf".to_string(), 1.0064_f64 * 3.0);

        let formatted = format_amounts(&amounts);

        let rendered = &formatted["tmAVMxFD4vAvu16Pefa2X66zixCWrYFpcUf"];
        assert_eq!(rendered, "3.01920000");
        assert!(rendered.split('.').nth(1).unwrap().len() <= 8);
    }

    #[test]
    fn preserves_clean_values() {
        let mut amounts = BTreeMap::new();
        amounts.insert("wallet-a".to_string(), 5.5_f64);

        let formatted = format_amounts(&amounts);

        assert_eq!(formatted["wallet-a"], "5.50000000");
    }
}
