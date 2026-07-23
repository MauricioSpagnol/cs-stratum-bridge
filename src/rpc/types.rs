//! Serde types for `csd` JSON-RPC responses (OPoI + wallet calls).

/// Result of `getopoistake`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OpoiStake {
    pub miner_address: String,
    /// "ACTIVE" | "UNSTAKING" | "RELEASED" | "SLASHED" | "SUSPENDED"
    pub status: String,
    pub last_renewal_height: Option<u64>,
    #[serde(default)]
    pub canary_strikes: u32,
}

/// Result of `stakeopoi`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StakeOpoiResult {
    pub txid: String,
    pub miner_address: String,
    pub amount: f64,
}

/// A pending or fetched OPoI inference request (`listopoirequests` / `getopoirequest`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OpoiRequest {
    pub request_id: String,
    pub model: String,
    pub prompt_hash: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub payment: f64,
    #[serde(default)]
    pub fee_per_token: f64,
    #[serde(default)]
    pub task_type: Option<String>,
    #[serde(default)]
    pub task_class: Option<String>,
}

/// Result of `submitopoiresponsecommit`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CommitResult {
    pub txid: String,
    pub nonce_hex: String,
    pub commit_window_closes_at_height: u64,
}

/// Result of `getmodelmanifest` — only the fields the shard pipeline needs
/// (arch_type/num_layers to size the pipeline, backbone_pom_root to verify
/// the GGUF a miner downloads, status to gate on ACTIVE). The daemon returns
/// several more fields (voting tallies, expert_pom_roots, etc.) that aren't
/// relevant here and are simply ignored by serde.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub arch_type: String,
    pub num_layers: u32,
    pub backbone_pom_root: String,
    /// "VOTING" | "APPROVED" | "ACTIVE" | "REJECTED"
    pub status: String,
}

/// One entry of a Model Execution Graph (`getmodelgraph`'s `shards` array).
/// `EXPERT` shards are present in the JSON for MoE/hybrid models but the
/// shard pipeline (Sessão 3, dense-only) skips any model whose graph
/// contains one — see `shard_engine.rs`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ShardDescriptor {
    pub shard_index: u32,
    /// "DENSE" | "EXPERT"
    pub shard_type: String,
    #[serde(default)]
    pub layer_start: Option<u32>,
    #[serde(default)]
    pub layer_end: Option<u32>,
    #[serde(default)]
    pub expert_id: Option<u32>,
}

/// Result of `getmodelgraph`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelGraph {
    pub model_id: String,
    pub shard_topology_hash: String,
    pub shards: Vec<ShardDescriptor>,
}

/// Result of `claimcoordinator`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClaimResult {
    pub txid: String,
    pub request_id: String,
    pub miner_address: String,
}

/// Result of `submitshardresult`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ShardResultTxRes {
    pub txid: String,
    pub request_id: String,
    pub shard_index: u32,
    pub miner_address: String,
}
