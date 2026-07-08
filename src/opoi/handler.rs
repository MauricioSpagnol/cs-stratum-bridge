//! Decouples the stratum proxy (src/proxy) from the OPoI engine
//! (src/opoi/engine.rs, built separately). The proxy only depends on this
//! trait; the engine implements it. This lets the proxy and the engine be
//! built/reviewed independently and wired together in main.rs.

use async_trait::async_trait;

use crate::error::AppError;
use crate::opoi::wire::OpoiSubmitResultParams;

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
