mod config;
mod error;

use config::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = Config::load();
    tracing::info!(
        listen_addr = %cfg.listen_addr,
        upstream_pool_addr = %cfg.upstream_pool_addr,
        opoi_address = %cfg.opoi_address,
        payout_address = %cfg.effective_payout_address(),
        "cs-stratum-bridge config loaded"
    );

    // Scaffold step: config + logging only. Proxy listener, csd RPC client,
    // DB, and the OPoI engine are added in subsequent steps.
    Ok(())
}
