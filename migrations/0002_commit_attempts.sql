-- Multi-identity stake pool support (see src/stake_pool.rs).
--
-- nOPoIVrfThreshold gates submitopoiresponse (REVEAL) at ~3% eligibility per
-- address on mainnet/testnet, and REVEAL must use the SAME address that
-- COMMITted (the daemon rejects a reveal whose miner_address has no
-- matching COMMIT). With a pool of N stake identities, the only way to
-- find out which one will be VRF-eligible at reveal time is to COMMIT with
-- every pool address for the same response (there's no read-only
-- eligibility check — see stake_pool.rs), then try revealing with each
-- once its own commit window closes, stopping at the first success.
--
-- opoi_commit_attempts tracks one row per (submission, pool address)
-- commit attempt. opoi_submissions.commit_txid/commit_nonce_hex/
-- commit_window_closes_at_height/reveal_txid keep recording only the
-- WINNING attempt's data (for backward-compatible reads of that table) —
-- this table is the source of truth for "which addresses were tried and
-- which one ultimately revealed."
CREATE TABLE opoi_commit_attempts (
    id                BIGSERIAL PRIMARY KEY,
    submission_id     BIGINT NOT NULL REFERENCES opoi_submissions(id),
    opoi_address      TEXT NOT NULL,
    status            VARCHAR(16) NOT NULL DEFAULT 'FAILED', -- COMMITTED | REVEALED | FAILED
    commit_txid       VARCHAR(128),
    commit_nonce_hex  VARCHAR(128),
    commit_window_closes_at_height INTEGER,
    reveal_txid       VARCHAR(128),
    fail_reason       TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One attempt row per (submission, address) — do_commit upserts on retry
-- instead of accumulating duplicates.
CREATE UNIQUE INDEX uq_opoi_commit_attempts_submission_address
    ON opoi_commit_attempts (submission_id, opoi_address);

CREATE INDEX idx_opoi_commit_attempts_submission_id ON opoi_commit_attempts (submission_id);

-- Backs poll_reveal_tick's "which attempts are ready to try revealing now" query.
CREATE INDEX idx_opoi_commit_attempts_ready
    ON opoi_commit_attempts (status, commit_window_closes_at_height)
    WHERE status = 'COMMITTED';
