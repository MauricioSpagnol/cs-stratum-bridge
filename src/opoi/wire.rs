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

// ── F15-H shard-pipeline messages (Sessão 3, Revisão v6) ─────────────────────
//
// Separate message types from OpoiAssign/OpoiSubmitResult above, not an
// extension of them: mixing shard and whole-model fields into one struct
// would force every shard-only field to be `Option<>` in a flow that never
// uses it (and vice versa), and the dispatch code would need an `if
// is_shard {...}` branch either way. Two self-contained pairs, two handlers.

/// Pushed downstream (bridge -> miner) as an `opoi.shard_assign`
/// notification. `model_gguf_url`/`model_gguf_sha256` (and the tokenizer
/// equivalents) are resolved by the bridge from `getmodelmanifest` +
/// `ModelSourceConfig` — the miner's `gguf_fetch.rs` never discovers a peer
/// or an expected hash on its own (see that module's doc comment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiShardAssign {
    pub request_id: String,
    /// Used purely as a local cache key by the miner's `gguf_fetch.rs`
    /// (`{model_id}.gguf`) so repeated requests for the same model reuse
    /// the download.
    pub model_id: String,
    pub shard_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    /// 0 = first token of the generation; >0 = a continuation step against
    /// an already-open KV-cache session (see cs-miner's `shard_session.rs`).
    pub token_index: u32,
    pub max_tokens: u32,
    pub input: ShardInputWire,
    pub model_gguf_url: String,
    pub model_gguf_sha256: String,
    /// `None` when `token_index > 0` or `layer_start > 0` — only the entry
    /// shard's first token ever needs to tokenize raw text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_tokenizer_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_tokenizer_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ShardInputWire {
    Prompt { prompt_hex: String },
    NextTokenId { token_id: u32 },
    /// F15-L: a speculative batch to verify in one forward pass — the
    /// entry-shard counterpart of `Tensor` below for later shards. Mirrors
    /// cs-miner's `pow/stratum.rs` byte-for-byte.
    TokenIds { token_ids: Vec<u32> },
    Tensor { shape: Vec<usize>, data_hex: String },
}

/// Received from a downstream miner as `opoi.shard_result`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiShardResult {
    pub request_id: String,
    pub shard_index: u32,
    pub token_index: u32,
    pub output_hash: String,
    pub output: ShardOutputWire,
    /// `Some` only when this shard's `layer_end == total_layers` (the last
    /// stage of the pipeline) — the argmax-decoded next token id, which the
    /// bridge feeds back into the entry shard as the next `token_index`'s
    /// `NextTokenId` input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_token_id: Option<u32>,
    /// F15-L: only populated (last shard, speculative batch) alongside
    /// `next_token_id` — the argmax at EVERY input position, letting the
    /// bridge run greedy rejection sampling across the whole verify batch
    /// instead of learning only the very last position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardOutputWire {
    pub shape: Vec<usize>,
    pub data_hex: String,
}

// ── F15-L speculative-decoding messages (Revisão v6) ─────────────────────
//
// Field names/shapes match cs-miner's `src/pow/stratum.rs` byte-for-byte —
// same shared-contract convention as the shard messages above.

/// Pushed downstream (bridge -> miner) as an `opoi.draft_generate`
/// notification: asks a `draft`-capable miner (F17-D) to run ITS OWN small
/// same-family model locally (no network round trip between steps) and
/// propose up to `k` candidate continuation tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiDraftGenerate {
    pub request_id: String,
    pub model_id: String,
    /// The draft model's OWN total layer count (it's a single-shard model —
    /// no splitting — but the miner's `SessionStore::step` contract still
    /// needs `total_layers` to know this call IS the last/only shard).
    pub total_layers: u32,
    pub k: u32,
    pub input: DraftInputWire,
    pub model_gguf_url: String,
    pub model_gguf_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_tokenizer_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_tokenizer_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum DraftInputWire {
    Prompt { prompt_hex: String },
    NextTokenId { token_id: u32 },
}

/// Received from a downstream miner as `opoi.draft_result`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiDraftResult {
    pub request_id: String,
    pub draft_token_ids: Vec<u32>,
}

/// Pushed downstream (bridge -> miner) as an `opoi.kv_rollback`
/// notification: after a speculative batch verify rejects a suffix of the
/// drafted tokens (or a round is fully accepted and needs to advance), roll
/// this session's KV-cache back to (or catch it up to) `keep_until_pos`
/// positions. `shard_index == DRAFT_SESSION_SHARD_INDEX` (`u32::MAX`, see
/// cs-miner's `shard_session.rs`) targets the draft-model session instead
/// of a real dense shard's — both share the miner's one `SessionStore`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiKvRollback {
    pub request_id: String,
    pub shard_index: u32,
    pub keep_until_pos: u32,
    /// When true, `keep_until_pos` is ignored: this session has reached
    /// `max_tokens` and should finalize (the same accumulated-hash
    /// commitment the non-speculative path always produced on the last
    /// token) and drop, instead of merely truncating.
    pub finalize: bool,
}

/// Received from a downstream miner as `opoi.kv_rollback_ack`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpoiKvRollbackAck {
    pub request_id: String,
    pub shard_index: u32,
    pub ok: bool,
    /// Only set when the request carried `finalize: true` and it
    /// succeeded — the true accumulated hash to submit on-chain (real
    /// shards only; a finalized draft session has no on-chain meaning and
    /// this is always `None` there).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finalized_hash: Option<String>,
}

/// Sentinel `shard_index` for a draft model's own session — mirrors
/// cs-miner's `shard_session::DRAFT_SESSION_SHARD_INDEX` byte-for-byte.
pub const DRAFT_SESSION_SHARD_INDEX: u32 = u32::MAX;

pub fn build_draft_generate_line(assign: &OpoiDraftGenerate) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "opoi.draft_generate",
        "params": [assign],
        "id": serde_json::Value::Null,
    });
    format!("{}\n", msg)
}

pub fn build_kv_rollback_line(rollback: &OpoiKvRollback) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "opoi.kv_rollback",
        "params": [rollback],
        "id": serde_json::Value::Null,
    });
    format!("{}\n", msg)
}

pub fn build_shard_assign_line(assign: &OpoiShardAssign) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "opoi.shard_assign",
        "params": [assign],
        "id": serde_json::Value::Null,
    });
    format!("{}\n", msg)
}

pub fn build_shard_result_ack(id: serde_json::Value) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "accepted": true },
        "error": serde_json::Value::Null,
    });
    format!("{}\n", msg)
}

pub fn build_shard_result_error(id: serde_json::Value, error_triple: serde_json::Value) -> String {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": serde_json::Value::Null,
        "error": error_triple,
    });
    format!("{}\n", msg)
}
