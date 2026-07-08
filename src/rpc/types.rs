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
