-- B3-lite (see src/opoi/b3lite.rs / b3lite_audit.rs, and "ESCOPO CONCRETO DA
-- B3-LITE" in CS COIN OPoI MELHOR IMPLEMENTAÇÃO.txt): served-response
-- receipts for the manifest-pinned (shard-routed) path, a random sample of
-- which get replayed through cs-miner's Auditor, plus the off-chain
-- consequences that follow a confirmed divergence. No on-chain/consensus
-- schema touched by any of this — see b3lite.rs's module doc.

CREATE TABLE b3lite_receipts (
    id                       BIGSERIAL PRIMARY KEY,
    request_id               VARCHAR(64) NOT NULL,
    miner_wallet             TEXT NOT NULL,
    model_id                 VARCHAR(128) NOT NULL,
    gguf_sha256              VARCHAR(64) NOT NULL,
    prompt_hash              VARCHAR(64),
    -- The raw prompt this request was served with, hex-encoded — persisted
    -- here (not just transiently cached, see shard_engine.rs's
    -- PipelineState::original_prompt_hex doc) because a later audit replay
    -- needs to re-tokenize the SAME prompt the miner was given.
    prompt_hex               TEXT NOT NULL,
    response_hash            VARCHAR(64) NOT NULL,
    -- Each generated token id as 4 bytes little-endian, concatenated in
    -- generation order — the SAME encoding shard_engine.rs's build_response
    -- already uses for response_hex, reused here for the identical reason.
    generated_token_ids_hex  TEXT NOT NULL,
    total_layers             INTEGER NOT NULL,
    signature_hex            TEXT NOT NULL,
    sampled                  BOOLEAN NOT NULL DEFAULT FALSE,
    -- NONE (not sampled) | PENDING (sampled, audit not run yet) |
    -- ADMISSIBLE | DIVERGENT | INCONCLUSIVE (audit subprocess itself
    -- failed — NOT itself a fraud signal, see auditor.rs's `audit` doc).
    audit_status             VARCHAR(16) NOT NULL DEFAULT 'NONE',
    audit_detail             TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    audited_at               TIMESTAMPTZ
);

CREATE INDEX idx_b3lite_receipts_request_id ON b3lite_receipts (request_id);
CREATE INDEX idx_b3lite_receipts_miner_wallet ON b3lite_receipts (miner_wallet);

-- Backs the audit_tick's "what's due for a real audit run" query.
CREATE INDEX idx_b3lite_receipts_pending_audit
    ON b3lite_receipts (sampled, audit_status)
    WHERE sampled = TRUE AND audit_status = 'PENDING';

CREATE TABLE b3lite_consequences (
    id            BIGSERIAL PRIMARY KEY,
    receipt_id    BIGINT NOT NULL REFERENCES b3lite_receipts(id),
    miner_wallet  TEXT NOT NULL,
    request_id    VARCHAR(64) NOT NULL,
    -- WITHHOLD_PAY (excludes this request_id from payout::payout_tick) |
    -- REPUTATION_FLAG (recorded only) | EJECTED (banned from future OPoI
    -- dispatch via miner_registry::MinerRegistry — PoW mining is
    -- unaffected, see MinerRegistry's ban doc).
    action        VARCHAR(24) NOT NULL,
    reason        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_b3lite_consequences_wallet ON b3lite_consequences (miner_wallet);
CREATE INDEX idx_b3lite_consequences_request_id ON b3lite_consequences (request_id);
