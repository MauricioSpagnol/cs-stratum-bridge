//! Wire-format types for the OPoI-specific stratum extension messages.
//! Field names/shapes match cs-miner's `src/pow/stratum.rs` byte-for-byte —
//! this is the shared contract between this bridge and the miner client.

use serde::{Deserialize, Serialize};

/// Pushed downstream (bridge -> miner) as a `opoi.assign` notification:
/// `{"jsonrpc":"2.0","method":"opoi.assign","params":[<this>],"id":null}`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiAssign {
    pub request_id: String,
    pub model: String,
    pub prompt_hex: String,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_class: Option<String>,
}

/// Received from a downstream miner as `opoi.submit_result`:
/// `{"jsonrpc":"2.0","id":<flagged_id>,"method":"opoi.submit_result","params":[<this>]}`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiSubmitResultParams {
    pub request_id: String,
    pub response_hash: String,
    pub response_hex: String,
    #[serde(default)]
    pub token_count: u32,
}

/// Builds the `opoi.assign` JSON-RPC notification line (newline-terminated,
/// ready to write to the downstream socket).
pub fn build_assign_line(assign: &OpoiAssign) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "opoi.assign",
        "params": [assign],
        "id": serde_json::Value::Null,
    });
    format!("{}\n", msg)
}

/// Builds a success response line for a received `opoi.submit_result`, id
/// echoed back exactly as received (including the OPOI_ID_FLAG bit cs-miner set).
pub fn build_submit_result_ack(id: serde_json::Value, submission_id: i64) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "accepted": true, "submission_id": submission_id },
        "error": serde_json::Value::Null,
    });
    format!("{}\n", msg)
}

/// Builds a rejection response line for a received `opoi.submit_result`.
pub fn build_submit_result_error(id: serde_json::Value, error_triple: serde_json::Value) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": serde_json::Value::Null,
        "error": error_triple,
    });
    format!("{}\n", msg)
}
