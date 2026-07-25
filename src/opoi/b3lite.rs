//! B3-lite receipt signing + audit sampling (2026-07-25 session — see
//! "ESCOPO CONCRETO DA B3-LITE" in `CS COIN OPoI MELHOR IMPLEMENTAÇÃO.txt`
//! at the monorepo root). Pure, DB-free pieces factored out of
//! `shard_engine.rs`/`b3lite_audit.rs` so they're testable without a live
//! `PgPool`, same pattern `shard_engine.rs`'s own `select_winning_wallet`/
//! `build_response` already use.
//!
//! What a "receipt" is for: B3-lite serves a manifest-pinned response
//! directly from one miner, off-chain, without waiting for the full
//! cross-pool on-chain commit/reveal/publish settlement to finish (see
//! `shard_engine.rs`'s `finalize_pipeline` — the response is already
//! recorded and fetchable via `GET /cscoin/opoi/{request_id}` well before
//! that settles). The receipt is this bridge operator's own record of
//! exactly what was served and to whom, signed so it can't be silently
//! altered after the fact, used for two things: (1) input to the sampling
//! decision below, and (2) the exact payload later replayed through
//! cs-miner's Auditor (`b3lite_audit.rs`) if sampled.
//!
//! Signing is HMAC-SHA256 over a canonical field encoding, keyed by an
//! operator-configured shared secret (`Config::b3lite_receipt_secret`) —
//! see `Cargo.toml`'s `hmac` dependency comment for why this is deliberately
//! symmetric, not a public/asymmetric signature.

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Bundles the two operator-config knobs `ShardEngine` needs to record
/// receipts, so its constructor takes one param instead of two and adding
/// a third B3-lite knob later doesn't mean touching every call site again.
/// `from_config` returns `None` when B3-lite is disabled (see
/// `Config::b3lite_enabled`) — `ShardEngine` treats that as "record no
/// receipts at all," not "record unsigned ones."
#[derive(Clone)]
pub struct B3LiteConfig {
    pub secret: Vec<u8>,
    pub sample_rate: f64,
}

impl B3LiteConfig {
    pub fn from_config(cfg: &crate::config::Config) -> Option<Self> {
        if !cfg.b3lite_enabled() {
            return None;
        }
        Some(Self { secret: cfg.b3lite_receipt_secret.as_bytes().to_vec(), sample_rate: cfg.b3lite_audit_sample_rate })
    }
}

/// Everything a B3-lite receipt commits to — see module doc. Token ids
/// (not text) because that's what the bridge's shard pipeline actually
/// produces (no on-bridge tokenizer — see `shard_engine.rs`'s
/// `finalize_pipeline` doc, simplification #1).
pub struct ReceiptFields<'a> {
    pub request_id: &'a str,
    pub miner_wallet: &'a str,
    pub model_id: &'a str,
    pub gguf_sha256: &'a str,
    pub response_hash: &'a str,
    pub generated_token_ids: &'a [u32],
}

/// Canonical byte encoding fed to the HMAC — every field length-prefixed
/// (as a `u32` little-endian count of items/bytes) so no ambiguity is
/// possible between e.g. `request_id`+`miner_wallet` concatenated one way
/// vs another (a classic HMAC-canonicalization pitfall: naive
/// concatenation lets an attacker shift a byte from one field into the
/// next and still forge a colliding message for a DIFFERENT logical
/// receipt).
fn canonical_bytes(fields: &ReceiptFields) -> Vec<u8> {
    let mut buf = Vec::new();
    for s in [fields.request_id, fields.miner_wallet, fields.model_id, fields.gguf_sha256, fields.response_hash] {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    buf.extend_from_slice(&(fields.generated_token_ids.len() as u32).to_le_bytes());
    for id in fields.generated_token_ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    buf
}

/// Signs `fields` with `secret`, returning the hex-encoded HMAC-SHA256 tag.
pub fn sign_receipt(secret: &[u8], fields: &ReceiptFields) -> String {
    // `new_from_slice` only fails for a key length HMAC-SHA256 rejects,
    // which never happens (HMAC accepts any key length, hashing it down
    // first if longer than the block size) — safe to unwrap.
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC-SHA256 accepts any key length");
    mac.update(&canonical_bytes(fields));
    hex::encode(mac.finalize().into_bytes())
}

/// Recomputes and compares — used by this module's own tests and available
/// for anyone holding `secret` to independently verify a stored receipt.
pub fn verify_receipt(secret: &[u8], fields: &ReceiptFields, signature_hex: &str) -> bool {
    sign_receipt(secret, fields).eq_ignore_ascii_case(signature_hex)
}

/// Deterministic (not RNG-based) sampling decision: hashes `request_id`
/// with the receipt's own signature as domain separation (so sampling
/// can't be predicted/gamed without already knowing the secret — a miner
/// guessing which of ITS OWN requests will be audited would need the same
/// secret the signature itself requires) and compares the first 8 bytes,
/// interpreted as a `u64`, against `rate` scaled to the full `u64` range.
/// `rate <= 0.0` never samples; `rate >= 1.0` always does.
pub fn should_sample(signature_hex: &str, rate: f64) -> bool {
    use sha2::Digest;
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    let digest = Sha256::digest(signature_hex.as_bytes());
    let first8: [u8; 8] = digest[..8].try_into().expect("sha256 digest is at least 8 bytes");
    let value = u64::from_be_bytes(first8);
    let threshold = (rate * u64::MAX as f64) as u64;
    value < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fields() -> ReceiptFields<'static> {
        ReceiptFields {
            request_id: "req-1",
            miner_wallet: "wallet-a",
            model_id: "QWEN2_5_0_5B",
            gguf_sha256: "aaaa",
            response_hash: "bbbb",
            generated_token_ids: &[1, 2, 3],
        }
    }

    #[test]
    fn sign_and_verify_round_trips() {
        let secret = b"test-secret";
        let fields = sample_fields();
        let sig = sign_receipt(secret, &fields);
        assert!(verify_receipt(secret, &fields, &sig));
    }

    #[test]
    fn verify_fails_with_wrong_secret() {
        let fields = sample_fields();
        let sig = sign_receipt(b"secret-a", &fields);
        assert!(!verify_receipt(b"secret-b", &fields, &sig));
    }

    #[test]
    fn verify_fails_if_token_ids_change() {
        let secret = b"test-secret";
        let fields = sample_fields();
        let sig = sign_receipt(secret, &fields);
        let mut tampered = sample_fields();
        let ids = [1u32, 2, 4]; // last token changed
        tampered.generated_token_ids = &ids;
        assert!(!verify_receipt(secret, &tampered, &sig));
    }

    #[test]
    fn verify_fails_if_response_hash_changes() {
        let secret = b"test-secret";
        let fields = sample_fields();
        let sig = sign_receipt(secret, &fields);
        let mut tampered = sample_fields();
        tampered.response_hash = "cccc";
        assert!(!verify_receipt(secret, &tampered, &sig));
    }

    #[test]
    fn field_boundary_shift_does_not_collide() {
        // Naive concatenation (no length prefixes) would make
        // ("ab", "c") and ("a", "bc") sign identically for
        // request_id/miner_wallet — this pins that the length-prefixed
        // canonical encoding tells them apart.
        let secret = b"test-secret";
        let a = ReceiptFields { request_id: "ab", miner_wallet: "c", model_id: "m", gguf_sha256: "g", response_hash: "r", generated_token_ids: &[] };
        let b = ReceiptFields { request_id: "a", miner_wallet: "bc", model_id: "m", gguf_sha256: "g", response_hash: "r", generated_token_ids: &[] };
        assert_ne!(sign_receipt(secret, &a), sign_receipt(secret, &b));
    }

    #[test]
    fn should_sample_never_samples_at_zero_rate() {
        for i in 0..50 {
            let sig = format!("sig-{i}");
            assert!(!should_sample(&sig, 0.0));
        }
    }

    #[test]
    fn should_sample_always_samples_at_one_rate() {
        for i in 0..50 {
            let sig = format!("sig-{i}");
            assert!(should_sample(&sig, 1.0));
        }
    }

    #[test]
    fn should_sample_is_deterministic_for_the_same_signature() {
        let sig = "some-signature-hex";
        let first = should_sample(sig, 0.5);
        for _ in 0..20 {
            assert_eq!(should_sample(sig, 0.5), first);
        }
    }

    /// Not a statistical proof, just a sanity check that a mid-range rate
    /// neither always nor never samples across a reasonably large,
    /// varied set of signatures.
    #[test]
    fn should_sample_at_moderate_rate_is_neither_always_nor_never() {
        let sampled = (0..2000).filter(|i| should_sample(&format!("sig-{i}"), 0.05)).count();
        assert!(sampled > 0, "expected at least some samples at 5% over 2000 draws");
        assert!(sampled < 2000, "expected NOT all draws sampled at 5%");
        // Loose bounds around the expected ~100 (5% of 2000) — this is a
        // hash-based deterministic draw, not a true RNG, so a wide margin
        // avoids a flaky test while still catching a badly broken scaling.
        assert!(sampled < 400, "5% rate sampled far too many: {sampled}/2000");
    }
}
