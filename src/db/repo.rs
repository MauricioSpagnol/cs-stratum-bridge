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

use super::models::{B3LiteReceipt, RevealCandidate, Submission};

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
///
/// Excludes any `request_id` under an active B3-lite `WITHHOLD_PAY`
/// consequence (see `insert_b3lite_consequence`/`b3lite_audit.rs`) — a
/// confirmed-divergent response stays unpaid indefinitely (no automatic
/// un-withhold: this is a manual-review off-chain consequence, on-chain
/// consensus/settlement for the request itself is untouched either way).
pub async fn list_unpaid_published(pool: &PgPool) -> Result<Vec<Submission>, AppError> {
    let submissions = sqlx::query_as::<_, Submission>(
        r#"
        SELECT * FROM opoi_submissions
        WHERE status = 'PUBLISHED' AND paid = false
          AND request_id NOT IN (SELECT request_id FROM b3lite_consequences WHERE action = 'WITHHOLD_PAY')
        "#,
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

/// Records (upserting on retry) that `opoi_address` successfully committed
/// on-chain for `submission_id` — one of possibly several pool addresses
/// tried in parallel by `do_commit` (see stake_pool.rs).
pub async fn upsert_commit_attempt_success(
    pool: &PgPool,
    submission_id: i64,
    opoi_address: &str,
    commit_txid: &str,
    nonce_hex: &str,
    closes_at_height: i32,
) -> Result<(), AppError> {
    sqlx::query(
        r#"
        INSERT INTO opoi_commit_attempts
            (submission_id, opoi_address, status, commit_txid, commit_nonce_hex, commit_window_closes_at_height)
        VALUES ($1, $2, 'COMMITTED', $3, $4, $5)
        ON CONFLICT (submission_id, opoi_address) DO UPDATE
        SET status = 'COMMITTED', commit_txid = $3, commit_nonce_hex = $4,
            commit_window_closes_at_height = $5, fail_reason = NULL, updated_at = NOW()
        "#,
    )
    .bind(submission_id)
    .bind(opoi_address)
    .bind(commit_txid)
    .bind(nonce_hex)
    .bind(closes_at_height)
    .execute(pool)
    .await?;

    Ok(())
}

/// Records (upserting on retry) that `opoi_address` failed to commit for
/// `submission_id` — e.g. that address's stake isn't ACTIVE, or the RPC
/// otherwise rejected it. Not a VRF-eligibility failure: COMMIT itself
/// isn't VRF-gated (only REVEAL is), so a commit failure means something
/// more basic is wrong with that address.
pub async fn upsert_commit_attempt_failure(
    pool: &PgPool,
    submission_id: i64,
    opoi_address: &str,
    reason: &str,
) -> Result<(), AppError> {
    let truncated: String = reason.chars().take(2000).collect();

    sqlx::query(
        r#"
        INSERT INTO opoi_commit_attempts (submission_id, opoi_address, status, fail_reason)
        VALUES ($1, $2, 'FAILED', $3)
        ON CONFLICT (submission_id, opoi_address) DO UPDATE
        SET status = 'FAILED', fail_reason = $3, updated_at = NOW()
        "#,
    )
    .bind(submission_id)
    .bind(opoi_address)
    .bind(truncated)
    .execute(pool)
    .await?;

    Ok(())
}

/// Attempts ready to try revealing right now: COMMITTED, their own window
/// already closed, belonging to a submission still at COMMITTED overall
/// (i.e. no other pool address has revealed it yet). Joined with the
/// parent submission's request_id/response_hash/token_count since REVEAL
/// needs all of it alongside the per-attempt nonce_hex.
pub async fn list_reveal_ready_candidates(pool: &PgPool, height: i64) -> Result<Vec<RevealCandidate>, AppError> {
    let rows = sqlx::query_as::<_, RevealCandidate>(
        r#"
        SELECT a.id AS attempt_id, a.submission_id, a.opoi_address,
               a.commit_nonce_hex AS nonce_hex,
               s.request_id, s.response_hash, s.token_count
        FROM opoi_commit_attempts a
        JOIN opoi_submissions s ON s.id = a.submission_id
        WHERE a.status = 'COMMITTED'
          AND a.commit_window_closes_at_height <= $1
          AND s.status = 'COMMITTED'
        "#,
    )
    .bind(height)
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// Marks one specific attempt as the one that revealed successfully.
pub async fn mark_attempt_revealed(pool: &PgPool, attempt_id: i64, reveal_txid: &str) -> Result<(), AppError> {
    sqlx::query(
        r#"
        UPDATE opoi_commit_attempts
        SET status = 'REVEALED', reveal_txid = $1, updated_at = NOW()
        WHERE id = $2
        "#,
    )
    .bind(reveal_txid)
    .bind(attempt_id)
    .execute(pool)
    .await?;

    Ok(())
}

// ---- B3-lite (see opoi/b3lite.rs / b3lite_audit.rs) ------------------------

/// Persists a served-response receipt — called once per manifest-pinned
/// (shard-routed) response, whether or not it ends up `sampled` (see
/// `b3lite::should_sample`). Returns the new row's id.
#[allow(clippy::too_many_arguments)]
pub async fn create_b3lite_receipt(
    pool: &PgPool,
    request_id: &str,
    miner_wallet: &str,
    model_id: &str,
    gguf_sha256: &str,
    prompt_hash: Option<&str>,
    prompt_hex: &str,
    response_hash: &str,
    generated_token_ids_hex: &str,
    total_layers: i32,
    signature_hex: &str,
    sampled: bool,
) -> Result<i64, AppError> {
    let audit_status = if sampled { "PENDING" } else { "NONE" };
    let row = sqlx::query(
        r#"
        INSERT INTO b3lite_receipts
            (request_id, miner_wallet, model_id, gguf_sha256, prompt_hash, prompt_hex,
             response_hash, generated_token_ids_hex, total_layers, signature_hex, sampled, audit_status)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id
        "#,
    )
    .bind(request_id)
    .bind(miner_wallet)
    .bind(model_id)
    .bind(gguf_sha256)
    .bind(prompt_hash)
    .bind(prompt_hex)
    .bind(response_hash)
    .bind(generated_token_ids_hex)
    .bind(total_layers)
    .bind(signature_hex)
    .bind(sampled)
    .bind(audit_status)
    .fetch_one(pool)
    .await?;

    Ok(row.try_get::<i64, _>("id")?)
}

/// Receipts sampled for audit whose audit hasn't run yet — what
/// `b3lite_audit::audit_tick` iterates.
pub async fn list_pending_b3lite_audits(pool: &PgPool) -> Result<Vec<B3LiteReceipt>, AppError> {
    let rows = sqlx::query_as::<_, B3LiteReceipt>(
        r#"SELECT * FROM b3lite_receipts WHERE sampled = TRUE AND audit_status = 'PENDING'"#,
    )
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// Records a completed audit run's outcome — `status` must be one of
/// `ADMISSIBLE` / `DIVERGENT` / `INCONCLUSIVE` (see migration doc).
pub async fn mark_b3lite_audit_result(
    pool: &PgPool,
    receipt_id: i64,
    status: &str,
    detail: &str,
) -> Result<(), AppError> {
    sqlx::query(
        r#"
        UPDATE b3lite_receipts
        SET audit_status = $1, audit_detail = $2, audited_at = NOW()
        WHERE id = $3
        "#,
    )
    .bind(status)
    .bind(detail)
    .bind(receipt_id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Appends an off-chain consequence row — see migration doc for the three
/// `action` values and `b3lite_audit.rs` for the policy that decides which
/// ones fire on a confirmed divergence.
pub async fn insert_b3lite_consequence(
    pool: &PgPool,
    receipt_id: i64,
    miner_wallet: &str,
    request_id: &str,
    action: &str,
    reason: &str,
) -> Result<(), AppError> {
    sqlx::query(
        r#"
        INSERT INTO b3lite_consequences (receipt_id, miner_wallet, request_id, action, reason)
        VALUES ($1, $2, $3, $4, $5)
        "#,
    )
    .bind(receipt_id)
    .bind(miner_wallet)
    .bind(request_id)
    .bind(action)
    .bind(reason)
    .execute(pool)
    .await?;

    Ok(())
}

/// Total `WITHHOLD_PAY` consequences ever recorded for `miner_wallet` —
/// used by the consequence policy to decide whether a repeat offender
/// should now also be `EJECTED` (see `b3lite_audit.rs`).
pub async fn count_withhold_consequences(pool: &PgPool, miner_wallet: &str) -> Result<i64, AppError> {
    let row = sqlx::query(
        r#"SELECT COUNT(*) AS n FROM b3lite_consequences WHERE miner_wallet = $1 AND action = 'WITHHOLD_PAY'"#,
    )
    .bind(miner_wallet)
    .fetch_one(pool)
    .await?;

    Ok(row.try_get::<i64, _>("n")?)
}

/// Every wallet with at least one durable `EJECTED` consequence row —
/// called once at startup (see `main.rs`) to rebuild
/// `MinerRegistry`'s in-memory `banned` set, which otherwise forgets every
/// ejection on a restart (see that field's doc comment).
pub async fn list_ejected_wallets(pool: &PgPool) -> Result<Vec<String>, AppError> {
    let rows =
        sqlx::query(r#"SELECT DISTINCT miner_wallet FROM b3lite_consequences WHERE action = 'EJECTED'"#).fetch_all(pool).await?;

    let mut wallets = Vec::with_capacity(rows.len());
    for row in rows {
        wallets.push(row.try_get::<String, _>("miner_wallet")?);
    }
    Ok(wallets)
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

#[cfg(test)]
mod b3lite_repo_tests {
    //! Live-DB smoke test for the B3-lite repo functions above — this
    //! module's own doc comment explains why these use the runtime-checked
    //! query API instead of compile-time-checked macros (no `DATABASE_URL`
    //! needed to COMPILE), but that also means nothing validates the SQL
    //! itself actually runs until something executes it against a real
    //! Postgres. Gated on `B3LITE_TEST_DATABASE_URL` (a throwaway/scratch
    //! database, migrated fresh — never point this at a real deployment's
    //! DB) so a normal `cargo test` with no such env var set just skips it
    //! rather than failing everyone else's run.
    use super::*;

    async fn test_pool() -> Option<PgPool> {
        let url = std::env::var("B3LITE_TEST_DATABASE_URL").ok()?;
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.expect("connect to test DB");
        sqlx::migrate!("./migrations").run(&pool).await.expect("run migrations against test DB");
        Some(pool)
    }

    #[tokio::test]
    async fn b3lite_receipt_and_consequence_lifecycle() {
        let Some(pool) = test_pool().await else {
            eprintln!("B3LITE_TEST_DATABASE_URL not set — skipping live-DB smoke test");
            return;
        };

        let request_id = format!("test-req-{}", uuid::Uuid::new_v4());
        let wallet = "test-wallet-a";

        // Not sampled — should never show up in list_pending_b3lite_audits.
        let unsampled_id = create_b3lite_receipt(
            &pool, &request_id, wallet, "QWEN2_5_0_5B", "deadbeef", Some("promptash"), "70726f6d7074",
            "resphash", "0100000002000000", 24, "sig-unsampled", false,
        ).await.expect("create unsampled receipt");

        let sampled_request_id = format!("test-req-{}", uuid::Uuid::new_v4());
        let sampled_id = create_b3lite_receipt(
            &pool, &sampled_request_id, wallet, "QWEN2_5_0_5B", "deadbeef", Some("promptash"), "70726f6d7074",
            "resphash2", "0300000004000000", 24, "sig-sampled", true,
        ).await.expect("create sampled receipt");

        let pending = list_pending_b3lite_audits(&pool).await.expect("list pending");
        assert!(pending.iter().any(|r| r.id == sampled_id), "sampled receipt should be pending audit");
        assert!(!pending.iter().any(|r| r.id == unsampled_id), "unsampled receipt should never be pending audit");

        let sampled_row = pending.iter().find(|r| r.id == sampled_id).unwrap();
        assert_eq!(sampled_row.audit_status, "PENDING");
        assert_eq!(sampled_row.request_id, sampled_request_id);
        assert_eq!(sampled_row.generated_token_ids_hex, "0300000004000000");
        assert!(sampled_row.sampled);

        mark_b3lite_audit_result(&pool, sampled_id, "DIVERGENT", "{\"positions\":[]}").await.expect("mark audit result");
        let pending_after = list_pending_b3lite_audits(&pool).await.expect("list pending after mark");
        assert!(!pending_after.iter().any(|r| r.id == sampled_id), "no longer pending once audited");

        insert_b3lite_consequence(&pool, sampled_id, wallet, &sampled_request_id, "WITHHOLD_PAY", "test divergence")
            .await
            .expect("insert consequence");
        let count = count_withhold_consequences(&pool, wallet).await.expect("count consequences");
        assert!(count >= 1);

        // Not yet EJECTED — only a single WITHHOLD_PAY so far.
        let ejected = list_ejected_wallets(&pool).await.expect("list ejected wallets");
        assert!(!ejected.contains(&wallet.to_string()), "one divergence alone shouldn't eject");

        insert_b3lite_consequence(&pool, sampled_id, wallet, &sampled_request_id, "EJECTED", "3 confirmed divergences")
            .await
            .expect("insert EJECTED consequence");
        let ejected = list_ejected_wallets(&pool).await.expect("list ejected wallets after EJECTED insert");
        assert!(ejected.contains(&wallet.to_string()), "wallet with an EJECTED row must be listed");

        // create_submission + list_unpaid_published: the withheld request_id
        // must be excluded even though it's PUBLISHED and unpaid.
        let submission_id = create_submission(
            &pool, wallet, &sampled_request_id, Some("QWEN2_5_0_5B"), Some("promptash"), None, None, "resphash2",
            "0300000004000000", 2,
        ).await.expect("create submission");
        mark_published(&pool, submission_id, 1.5).await.expect("mark published");

        let unpaid = list_unpaid_published(&pool).await.expect("list unpaid published");
        assert!(
            !unpaid.iter().any(|s| s.request_id == sampled_request_id),
            "a request_id under an active WITHHOLD_PAY consequence must be excluded from payout"
        );

        // A DIFFERENT, non-withheld request should still show up normally.
        let other_request_id = format!("test-req-{}", uuid::Uuid::new_v4());
        let other_submission_id = create_submission(
            &pool, wallet, &other_request_id, Some("QWEN2_5_0_5B"), Some("promptash"), None, None, "resphash3",
            "0500000006000000", 2,
        ).await.expect("create other submission");
        mark_published(&pool, other_submission_id, 1.5).await.expect("mark other published");
        let unpaid_after = list_unpaid_published(&pool).await.expect("list unpaid published again");
        assert!(
            unpaid_after.iter().any(|s| s.request_id == other_request_id),
            "a non-withheld published+unpaid request should still be returned"
        );
    }
}
