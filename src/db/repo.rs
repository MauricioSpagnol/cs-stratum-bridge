//! Persistence operations for OPoI submissions and stake events.
//!
//! Deliberately uses the runtime-checked `sqlx::query()` / `sqlx::query_as()`
//! builder API rather than the `query!()` / `query_as!()` compile-time macros.
//! Those macros require either a live `DATABASE_URL` connection or a
//! prepared `.sqlx` offline-cache directory at compile time, and neither
//! exists yet in this fresh scaffold — using them here would break the
//! build for whoever compiles this next.

use sqlx::{PgPool, Row};

use crate::error::AppError;

use super::models::Submission;

/// Inserts a new submission row with `status = 'RECEIVED'` and returns its
/// generated id.
pub async fn create_submission(
    pool: &PgPool,
    miner_wallet: &str,
    request_id: &str,
    model: Option<&str>,
    prompt_hash: Option<&str>,
    payment_base: Option<f64>,
    fee_per_token: Option<f64>,
    response_hash: &str,
    response_hex: &str,
    token_count: i32,
) -> Result<i64, AppError> {
    let row = sqlx::query(
        r#"
        INSERT INTO opoi_submissions
            (miner_wallet, request_id, model, prompt_hash, payment_base,
             fee_per_token, response_hash, response_hex, token_count, status)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'RECEIVED')
        RETURNING id
        "#,
    )
    .bind(miner_wallet)
    .bind(request_id)
    .bind(model)
    .bind(prompt_hash)
    .bind(payment_base)
    .bind(fee_per_token)
    .bind(response_hash)
    .bind(response_hex)
    .bind(token_count)
    .fetch_one(pool)
    .await?;

    Ok(row.try_get::<i64, _>("id")?)
}

/// Finds the currently-active (non-FAILED) submission for a given
/// `request_id`, matching the partial unique index
/// `uq_opoi_submissions_active_request`.
pub async fn find_active_by_request_id(
    pool: &PgPool,
    request_id: &str,
) -> Result<Option<Submission>, AppError> {
    let submission = sqlx::query_as::<_, Submission>(
        r#"
        SELECT * FROM opoi_submissions
        WHERE request_id = $1 AND status <> 'FAILED'
        "#,
    )
    .bind(request_id)
    .fetch_optional(pool)
    .await?;

    Ok(submission)
}

/// Marks a submission as committed on-chain.
pub async fn mark_committed(
    pool: &PgPool,
    id: i64,
    commit_txid: &str,
    nonce_hex: &str,
    closes_at_height: i32,
) -> Result<(), AppError> {
    sqlx::query(
        r#"
        UPDATE opoi_submissions
        SET status = 'COMMITTED',
            commit_txid = $1,
            commit_nonce_hex = $2,
            commit_window_closes_at_height = $3,
            updated_at = NOW()
        WHERE id = $4
        "#,
    )
    .bind(commit_txid)
    .bind(nonce_hex)
    .bind(closes_at_height)
    .bind(id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Marks a submission as revealed on-chain.
pub async fn mark_revealed(pool: &PgPool, id: i64, reveal_txid: &str) -> Result<(), AppError> {
    sqlx::query(
        r#"
        UPDATE opoi_submissions
        SET status = 'REVEALED',
            reveal_txid = $1,
            updated_at = NOW()
        WHERE id = $2
        "#,
    )
    .bind(reveal_txid)
    .bind(id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Marks a submission as published, recording the reward amount that was
/// determined for it.
pub async fn mark_published(pool: &PgPool, id: i64, reward_amount: f64) -> Result<(), AppError> {
    sqlx::query(
        r#"
        UPDATE opoi_submissions
        SET status = 'PUBLISHED',
            content_published = true,
            reward_amount = $1,
            updated_at = NOW()
        WHERE id = $2
        "#,
    )
    .bind(reward_amount)
    .bind(id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Marks a submission as failed, storing a truncated failure reason.
///
/// `reason` is truncated to at most 2000 chars using a `chars()`-based take
/// rather than byte slicing, so multi-byte UTF-8 reasons can't be cut on a
/// non-char boundary and panic.
pub async fn mark_failed(pool: &PgPool, id: i64, reason: &str) -> Result<(), AppError> {
    let truncated: String = reason.chars().take(2000).collect();

    sqlx::query(
        r#"
        UPDATE opoi_submissions
        SET status = 'FAILED',
            fail_reason = $1,
            updated_at = NOW()
        WHERE id = $2
        "#,
    )
    .bind(truncated)
    .bind(id)
    .execute(pool)
    .await?;

    Ok(())
}

/// The startup-recovery query — must be called on boot, unlike the Node.js
/// original this replaces which defined an equivalent query but never
/// called it. Returns every submission still sitting at `RECEIVED` so the
/// service can resume driving them through commit/reveal/publish after a
/// restart.
pub async fn list_received(pool: &PgPool) -> Result<Vec<Submission>, AppError> {
    let submissions = sqlx::query_as::<_, Submission>(
        r#"SELECT * FROM opoi_submissions WHERE status = 'RECEIVED'"#,
    )
    .fetch_all(pool)
    .await?;

    Ok(submissions)
}

/// Returns every submission currently sitting at `COMMITTED`, e.g. so the
/// service can resume waiting on / driving their reveal step after a
/// restart.
pub async fn list_committed(pool: &PgPool) -> Result<Vec<Submission>, AppError> {
    let submissions = sqlx::query_as::<_, Submission>(
        r#"SELECT * FROM opoi_submissions WHERE status = 'COMMITTED'"#,
    )
    .fetch_all(pool)
    .await?;

    Ok(submissions)
}

/// Returns every submission that has been revealed on-chain but whose
/// content has not yet been published.
pub async fn list_revealed_unpublished(pool: &PgPool) -> Result<Vec<Submission>, AppError> {
    let submissions = sqlx::query_as::<_, Submission>(
        r#"SELECT * FROM opoi_submissions WHERE status = 'REVEALED' AND content_published = false"#,
    )
    .fetch_all(pool)
    .await?;

    Ok(submissions)
}

/// Returns every published, not-yet-paid submission — used by the payout
/// loop, which needs both the amount and the request_id (to mark exactly
/// the rows that were folded into a payout tx as paid) per miner wallet.
pub async fn list_unpaid_published(pool: &PgPool) -> Result<Vec<Submission>, AppError> {
    let submissions = sqlx::query_as::<_, Submission>(
        r#"SELECT * FROM opoi_submissions WHERE status = 'PUBLISHED' AND paid = false"#,
    )
    .fetch_all(pool)
    .await?;

    Ok(submissions)
}

/// Returns the total unpaid, published reward balance per miner wallet
/// (wallets with a zero or negative outstanding balance are excluded).
/// Convenience/read-only aggregate (e.g. for a status endpoint) — the
/// payout loop itself uses `list_unpaid_published` since it also needs
/// individual request_ids to mark as paid.
pub async fn unpaid_balance_by_wallet(pool: &PgPool) -> Result<Vec<(String, f64)>, AppError> {
    let rows = sqlx::query(
        r#"
        SELECT miner_wallet, SUM(reward_amount) AS total
        FROM opoi_submissions
        WHERE status = 'PUBLISHED' AND paid = false
        GROUP BY miner_wallet
        HAVING SUM(reward_amount) > 0
        "#,
    )
    .fetch_all(pool)
    .await?;

    let mut balances = Vec::with_capacity(rows.len());
    for row in rows {
        let miner_wallet: String = row.try_get("miner_wallet")?;
        let total: f64 = row.try_get("total")?;
        balances.push((miner_wallet, total));
    }

    Ok(balances)
}

/// Bulk-marks every submission that contributed to one payout transaction
/// as paid.
pub async fn mark_paid(
    pool: &PgPool,
    request_ids: &[String],
    payout_txid: &str,
) -> Result<(), AppError> {
    sqlx::query(
        r#"
        UPDATE opoi_submissions
        SET paid = true,
            payout_txid = $1,
            updated_at = NOW()
        WHERE request_id = ANY($2)
        "#,
    )
    .bind(payout_txid)
    .bind(request_ids)
    .execute(pool)
    .await?;

    Ok(())
}

/// Appends an entry to the stake-event audit log.
pub async fn log_stake_event(
    pool: &PgPool,
    event_type: &str,
    txid: Option<&str>,
    detail: &str,
) -> Result<(), AppError> {
    sqlx::query(
        r#"
        INSERT INTO opoi_stake_events (event_type, txid, detail)
        VALUES ($1, $2, $3)
        "#,
    )
    .bind(event_type)
    .bind(txid)
    .bind(detail)
    .execute(pool)
    .await?;

    Ok(())
}
