//! Row types and enums for the `opoi_submissions` / `opoi_stake_events` tables.

use chrono::{DateTime, Utc};

/// Lifecycle status of an OPoI submission.
///
/// Mirrors the `status` column of `opoi_submissions`, which is stored as
/// plain uppercase text (not a Postgres enum type) so that ad-hoc SQL /
/// dashboards can filter on it without needing to know about a custom type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionStatus {
    Received,
    Committed,
    Revealed,
    Published,
    Failed,
}

impl SubmissionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SubmissionStatus::Received => "RECEIVED",
            SubmissionStatus::Committed => "COMMITTED",
            SubmissionStatus::Revealed => "REVEALED",
            SubmissionStatus::Published => "PUBLISHED",
            SubmissionStatus::Failed => "FAILED",
        }
    }

    /// Parses a status string as stored in the DB.
    ///
    /// This is an internal invariant (the DB only ever contains values this
    /// service itself wrote), not user input, so an unrecognized value
    /// indicates data corruption or a schema/code drift bug and panics
    /// rather than returning a `Result`.
    pub fn from_str(s: &str) -> Self {
        match s {
            "RECEIVED" => SubmissionStatus::Received,
            "COMMITTED" => SubmissionStatus::Committed,
            "REVEALED" => SubmissionStatus::Revealed,
            "PUBLISHED" => SubmissionStatus::Published,
            "FAILED" => SubmissionStatus::Failed,
            other => panic!("unrecognized opoi_submissions.status value: {other:?}"),
        }
    }
}

// NOTE: this crate's sqlx feature set (see Cargo.toml: "runtime-tokio-rustls",
// "postgres", "chrono", "macros", "migrate") does not enable the "bigdecimal"
// or "rust_decimal" features, so the NUMERIC(18,8) columns are mapped to f64
// here rather than sqlx::types::BigDecimal to avoid a missing-feature compile
// error.
#[derive(Debug, Clone, sqlx::FromRow, serde::Serialize)]
pub struct Submission {
    pub id: i64,
    pub miner_wallet: String,
    pub request_id: String,
    pub model: Option<String>,
    pub prompt_hash: Option<String>,
    pub payment_base: Option<f64>,
    pub fee_per_token: Option<f64>,
    pub response_hash: String,
    pub response_hex: Option<String>,
    pub token_count: i32,
    pub status: String,
    pub fail_reason: Option<String>,
    pub commit_txid: Option<String>,
    pub commit_nonce_hex: Option<String>,
    pub commit_window_closes_at_height: Option<i32>,
    pub reveal_txid: Option<String>,
    pub content_published: bool,
    pub reward_amount: Option<f64>,
    pub paid: bool,
    pub payout_txid: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A commit attempt (see `stake_pool.rs`/migration `0002_commit_attempts.sql`
/// for why a submission can have more than one) joined with the fields of
/// its parent submission needed to attempt a reveal — what
/// `poll_reveal_tick` actually iterates.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct RevealCandidate {
    pub attempt_id: i64,
    pub submission_id: i64,
    pub opoi_address: String,
    pub nonce_hex: String,
    pub request_id: String,
    pub response_hash: String,
    pub token_count: i32,
}
