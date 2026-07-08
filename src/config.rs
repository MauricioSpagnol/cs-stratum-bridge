use clap::Parser;

/// cs-stratum-bridge — bridge/adapter between stratum-speaking cs-miner
/// clients and the csd daemon. Relays PoW `mining.*` traffic through to the
/// real upstream pool unchanged; owns the OPoI protocol itself.
#[derive(Parser, Debug, Clone)]
#[command(name = "cs-stratum-bridge")]
pub struct Config {
    /// Address the bridge listens on for downstream cs-miner connections.
    #[arg(long, env = "LISTEN_ADDR", default_value = "0.0.0.0:3532")]
    pub listen_addr: String,

    /// Real upstream pool (back-pool) address — all mining.* traffic relays here.
    #[arg(long, env = "UPSTREAM_POOL_ADDR")]
    pub upstream_pool_addr: String,

    /// csd JSON-RPC base URL, e.g. http://127.0.0.1:26124
    #[arg(long, env = "CSD_RPC_URL")]
    pub csd_rpc_url: String,

    #[arg(long, env = "CSD_RPC_USER")]
    pub csd_rpc_user: String,

    #[arg(long, env = "CSD_RPC_PASS")]
    pub csd_rpc_pass: String,

    /// Postgres connection string, dedicated database for this service.
    #[arg(long, env = "DATABASE_URL")]
    pub database_url: String,

    /// The bridge's own OPoI stake identity. Its private key must already be
    /// imported in csd's own wallet — this service never holds key material.
    #[arg(long, env = "OPOI_ADDRESS")]
    pub opoi_address: String,

    #[arg(long, env = "OPOI_COLLATERAL_TXID", default_value = "")]
    pub opoi_collateral_txid: String,

    #[arg(long, env = "OPOI_COLLATERAL_VOUT", default_value_t = 0)]
    pub opoi_collateral_vout: u32,

    #[arg(long, env = "OPOI_ENDPOINT", default_value = "")]
    pub opoi_endpoint: String,

    #[arg(long, env = "OPOI_MODEL_ID", default_value = "")]
    pub opoi_model_id: String,

    /// Address miner payouts are sent from. Defaults to opoi_address if unset.
    /// If different, its key must also be imported in csd's wallet.
    #[arg(long, env = "PAYOUT_ADDRESS", default_value = "")]
    pub payout_address: String,

    #[arg(long, env = "POOL_FEE_PERCENT", default_value_t = 0.0)]
    pub pool_fee_percent: f64,

    /// Minimum unpaid balance (in CS) before a miner is included in a payout run.
    #[arg(long, env = "MIN_PAYOUT_CS", default_value_t = 1.0)]
    pub min_payout_cs: f64,

    /// Address the requester-facing HTTP API (prompt intake, status) listens on.
    #[arg(long, env = "HTTP_LISTEN_ADDR", default_value = "0.0.0.0:3533")]
    pub http_listen_addr: String,

    /// Shared-secret required on POST /cscoin/opoi/prompt via the x-opoi-api-key header.
    #[arg(long, env = "OPOI_REQUESTER_API_KEY")]
    pub opoi_requester_api_key: String,

    #[arg(long, env = "POLL_INTERVAL_MS", default_value_t = 15_000)]
    pub poll_interval_ms: u64,

    #[arg(long, env = "REVEAL_POLL_INTERVAL_MS", default_value_t = 15_000)]
    pub reveal_poll_interval_ms: u64,

    #[arg(long, env = "RENEW_INTERVAL_MS", default_value_t = 21_600_000)]
    pub renew_interval_ms: u64,

    #[arg(long, env = "PAYOUT_INTERVAL_MS", default_value_t = 3_600_000)]
    pub payout_interval_ms: u64,
}

impl Config {
    /// Address that should sign/receive miner payouts — payout_address if
    /// set, otherwise falls back to the bridge's own OPoI stake address.
    pub fn effective_payout_address(&self) -> &str {
        if self.payout_address.is_empty() {
            &self.opoi_address
        } else {
            &self.payout_address
        }
    }

    pub fn load() -> Self {
        // Load .env before clap reads the process environment, mirroring the
        // Node/back-pool convention — but .env is gitignored from commit 1,
        // never carries real secrets into version control.
        dotenvy::dotenv().ok();
        Config::parse()
    }
}
