//! TCP accept loop for the stratum proxy. Binds `listen_addr`, and for every
//! downstream (cs-miner) connection that comes in, spawns an independent
//! session that dials the real upstream pool and relays traffic. A single
//! connection erroring or panicking can never take the listener down.

use std::sync::Arc;

use tokio::net::TcpListener;

use crate::miner_registry::MinerRegistry;
use crate::opoi::handler::{AuditHandler, OpoiHandler, ShardHandler, SpeculativeHandler};
use crate::proxy::session;

pub async fn run(
    listen_addr: String,
    upstream_addr: String,
    registry: Arc<MinerRegistry>,
    handler: Arc<dyn OpoiHandler>,
    shard_handler: Arc<dyn ShardHandler>,
    speculative_handler: Arc<dyn SpeculativeHandler>,
    audit_handler: Arc<dyn AuditHandler>,
) -> anyhow::Result<()> {
    // Owned String (not &str): this future is passed to tokio::spawn from
    // main.rs and must be 'static, which rules out borrowing from Config.
    let listener = TcpListener::bind(&listen_addr).await?;
    tracing::info!(listen_addr = %listen_addr, "stratum proxy listening");

    loop {
        let (socket, peer_addr) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                // A transient accept error must not kill the loop.
                tracing::warn!(error = %e, "failed to accept downstream connection; continuing");
                continue;
            }
        };

        tracing::info!(peer = %peer_addr, "miner connected");

        let upstream_addr = upstream_addr.clone();
        let registry = registry.clone();
        let handler = handler.clone();
        let shard_handler = shard_handler.clone();
        let speculative_handler = speculative_handler.clone();
        let audit_handler = audit_handler.clone();

        tokio::spawn(async move {
            match session::run_session(socket, upstream_addr, registry, handler, shard_handler, speculative_handler, audit_handler).await {
                Ok(()) => {
                    tracing::info!(peer = %peer_addr, "miner disconnected");
                }
                Err(e) => {
                    tracing::warn!(peer = %peer_addr, error = %e, "session ended with error");
                }
            }
        });
        // Note: tokio::spawn isolates panics inside the session task from
        // this accept loop — we don't await the JoinHandle here, so even a
        // panicking session can never propagate out and take the listener
        // down.
    }
}
