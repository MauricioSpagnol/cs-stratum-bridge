//! Decouples the stratum proxy (src/proxy) from the OPoI engine
//! (src/opoi/engine.rs, built separately). The proxy only depends on this
//! trait; the engine implements it. This lets the proxy and the engine be
//! built/reviewed independently and wired together in main.rs.

use async_trait::async_trait;

use crate::error::AppError;
use crate::opoi::wire::{OpoiAuditResult, OpoiDraftResult, OpoiKvRollbackAck, OpoiShardResult, OpoiSubmitResultParams};

#[async_trait]
pub trait OpoiHandler: Send + Sync {
    /// Called when an authorized downstream connection sends `opoi.submit_result`.
    /// `wallet` is the address that connection authorized with (mining.authorize,
    /// already confirmed accepted by the upstream pool). Returns a submission id
    /// on success (used in the ACK), or an AppError describing why it was rejected
    /// (unauthorized / unknown request / not assigned to this wallet / etc).
    async fn handle_submit_result(
        &self,
        wallet: &str,
        params: OpoiSubmitResultParams,
    ) -> Result<i64, AppError>;

    /// Called by the proxy session when a previously-authorized connection
    /// disconnects, so any request_id assigned to it (but never answered)
    /// can be released back to the pool for reassignment on the next
    /// poll_and_assign_tick, instead of staying stuck forever.
    async fn on_disconnect(&self, wallet: &str);
}

/// F15-H (Sessão 3): the shard-pipeline counterpart of `OpoiHandler`, kept
/// as a SEPARATE trait rather than added to `OpoiHandler` — `ShardEngine`
/// is a distinct component from `OpoiEngine` (see shard_engine.rs's module
/// doc), and mixing the two concerns into one trait would force `OpoiEngine`
/// to implement a method it has no business handling.
#[async_trait]
pub trait ShardHandler: Send + Sync {
    /// Called when an authorized downstream connection sends `opoi.shard_result`.
    async fn handle_shard_result(&self, wallet: &str, result: OpoiShardResult) -> Result<(), AppError>;

    /// Mirrors `OpoiHandler::on_disconnect` — releases any shard assignment
    /// the disconnecting wallet was holding.
    async fn on_disconnect(&self, wallet: &str);
}

/// F15-L: the speculative-decoding counterpart of `ShardHandler` — kept as
/// its own trait, same reasoning as `ShardHandler` vs `OpoiHandler`
/// (`SpeculativeEngine` is a distinct component, see
/// `speculative_engine.rs`'s module doc). Neither inbound message this
/// trait handles gets an explicit reply line sent back down the wire (see
/// `proxy/session.rs`'s interception of `opoi.draft_result` /
/// `opoi.kv_rollback_ack` for why: cs-miner's own `pow/stratum.rs` has no
/// response-handling branch for either ack's tagged request id — it only
/// logs having sent them — so a reply here would either go unread or, worse,
/// be misread by the generic `id >= 3` PoW-submit-ack branch).
#[async_trait]
pub trait SpeculativeHandler: Send + Sync {
    /// Called when an authorized downstream connection sends
    /// `opoi.draft_result` (the reply to a previously-pushed
    /// `opoi.draft_generate`).
    async fn handle_draft_result(&self, wallet: &str, result: OpoiDraftResult);

    /// Called when an authorized downstream connection sends
    /// `opoi.kv_rollback_ack` (the reply to a previously-pushed
    /// `opoi.kv_rollback`).
    async fn handle_kv_rollback_ack(&self, wallet: &str, ack: OpoiKvRollbackAck);

    /// Mirrors `ShardHandler::on_disconnect`.
    async fn on_disconnect(&self, wallet: &str);
}

/// B3-lite (2026-07-25 session): the audit-dispatch counterpart of
/// `ShardHandler` — kept as its own trait, same reasoning as the others
/// (`B3LiteAuditor` is a distinct component, see `b3lite_audit.rs`'s
/// module doc).
#[async_trait]
pub trait AuditHandler: Send + Sync {
    /// Called when an authorized downstream connection sends
    /// `opoi.audit_result` (the reply to a previously-pushed
    /// `opoi.audit_assign`). Returns an `AppError` if this wallet wasn't
    /// the one dispatched to for `result.request_id` (or no dispatch is
    /// outstanding for it at all) — same "assignment verification" contract
    /// `ShardHandler::handle_shard_result` has.
    async fn handle_audit_result(&self, wallet: &str, result: OpoiAuditResult) -> Result<(), AppError>;

    /// Mirrors `ShardHandler::on_disconnect`.
    async fn on_disconnect(&self, wallet: &str);
}
