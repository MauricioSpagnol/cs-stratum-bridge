-- Initial schema for the OPoI pool-side bridge persistence layer.
--
-- opoi_submissions tracks each miner-submitted response through its full
-- lifecycle: RECEIVED -> COMMITTED -> REVEALED -> PUBLISHED (or FAILED at
-- any point), plus payout bookkeeping (paid / payout_txid) once a reward
-- has been swept into a batched payout transaction.
--
-- opoi_stake_events is an append-only audit log of pool-side stake/renew
-- actions taken against the chain (and their failures).

CREATE TABLE opoi_submissions (
    id                BIGSERIAL PRIMARY KEY,
    miner_wallet      TEXT NOT NULL,
    request_id        VARCHAR(64) NOT NULL,
    model             VARCHAR(128),
    prompt_hash       VARCHAR(64),
    -- DOUBLE PRECISION (not NUMERIC): this crate's sqlx feature set has
    -- neither "bigdecimal" nor "rust_decimal" enabled, so these are decoded
    -- as f64 in Rust (see src/db/models.rs) — NUMERIC would fail to decode
    -- at runtime without one of those features. Also consistent with the
    -- rest of the codebase, which already treats every on-chain amount
    -- (payment, fee_per_token, stake amount) as f64 straight from the
    -- daemon's JSON-RPC responses.
    payment_base      DOUBLE PRECISION,
    fee_per_token     DOUBLE PRECISION,
    response_hash     VARCHAR(64) NOT NULL,
    response_hex      TEXT,
    token_count       INTEGER NOT NULL DEFAULT 0,
    status            VARCHAR(16) NOT NULL DEFAULT 'RECEIVED',
    fail_reason       TEXT,
    commit_txid       VARCHAR(128),
    commit_nonce_hex  VARCHAR(128),
    commit_window_closes_at_height INTEGER,
    reveal_txid       VARCHAR(128),
    content_published BOOLEAN NOT NULL DEFAULT FALSE,
    reward_amount     DOUBLE PRECISION,
    paid              BOOLEAN NOT NULL DEFAULT FALSE,
    payout_txid       VARCHAR(128),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_opoi_submissions_request_id ON opoi_submissions (request_id);
CREATE INDEX idx_opoi_submissions_status ON opoi_submissions (status);
CREATE INDEX idx_opoi_submissions_miner_wallet ON opoi_submissions (miner_wallet);

-- Only one non-FAILED submission may exist per request_id at a time; a
-- request may be retried (and thus resubmitted) after a prior attempt
-- fails, which is why FAILED rows are excluded from the uniqueness check.
CREATE UNIQUE INDEX uq_opoi_submissions_active_request
    ON opoi_submissions (request_id) WHERE status <> 'FAILED';

CREATE TABLE opoi_stake_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(16) NOT NULL, -- STAKE | RENEW | STAKE_ERROR | RENEW_ERROR
    txid VARCHAR(128),
    detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
