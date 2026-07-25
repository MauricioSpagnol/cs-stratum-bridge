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

    /// Comma-separated list of the bridge's OPoI stake identities. Each
    /// address's private key must already be imported in csd's own wallet
    /// (this service never holds key material) and, beyond the first/
    /// primary address, must already have an ACTIVE stake created
    /// out-of-band (`stakeopoi`) — OPOI_COLLATERAL_TXID/_VOUT below only
    /// auto-bootstrap the primary one. Multiple identities exist purely to
    /// raise the odds of VRF eligibility: nOPoIVrfThreshold/
    /// nOPoICoordinatorThreshold/nOPoIShardThreshold gate every reveal/
    /// coordinator-claim/shard-submit at ~3% per address on both mainnet
    /// and testnet (only regtest relaxes this) — see StakePool.
    #[arg(long, env = "OPOI_ADDRESSES")]
    pub opoi_addresses: String,

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

    /// B3-lite (see opoi/b3lite.rs): HMAC-SHA256 key for signing served-
    /// response receipts. Empty (the default) means B3-lite receipts are
    /// unsigned/disabled — `Config::b3lite_enabled` gates on this.
    #[arg(long, env = "B3LITE_RECEIPT_SECRET", default_value = "")]
    pub b3lite_receipt_secret: String,

    /// Fraction (0.0-1.0) of manifest-pinned (shard-routed) served
    /// responses randomly selected for a real Auditor replay — see
    /// `opoi/b3lite.rs`'s `should_sample` for the deterministic (hash-based,
    /// not RNG-based) selection rule.
    #[arg(long, env = "B3LITE_AUDIT_SAMPLE_RATE", default_value_t = 0.05)]
    pub b3lite_audit_sample_rate: f64,

    /// Path to a `cs-miner` binary this bridge invokes as a LOCAL subprocess
    /// to run the Auditor replay (`cs-miner --audit-request <file>`) — see
    /// `opoi/b3lite_audit.rs`. This is the fallback path, always available:
    /// `AUDITOR_TRUSTED_WALLETS` below is the preferred path when a trusted
    /// wallet happens to be connected. Both paths share the same trust
    /// boundary (an audit is only ever run by THIS operator or a wallet
    /// THIS operator explicitly named as trusted) — dispatching to an
    /// arbitrary connected miner instead would add no real security (see
    /// `b3lite_audit.rs`'s module doc).
    #[arg(long, env = "AUDITOR_CS_MINER_BIN", default_value = "cs-miner")]
    pub auditor_cs_miner_bin: String,

    /// Scratch directory the Auditor subprocess downloads/caches GGUFs and
    /// tokenizers into (its own `gguf_fetch` cache — separate from any
    /// miner's own cache dir).
    #[arg(long, env = "AUDITOR_CACHE_DIR", default_value = "./auditor_cache")]
    pub auditor_cache_dir: String,

    #[arg(long, env = "B3LITE_AUDIT_POLL_INTERVAL_MS", default_value_t = 30_000)]
    pub b3lite_audit_poll_interval_ms: u64,

    /// Comma-separated wallet addresses this operator trusts to run a real
    /// audit replay over stratum (`opoi.audit_assign`/`opoi.audit_result`)
    /// instead of (or before falling back to) the local subprocess — see
    /// `auditor_cs_miner_bin`'s doc for why this doesn't weaken the trust
    /// model: dispatch only ever goes to a wallet in THIS list, never to an
    /// arbitrary connected miner. Empty (the default) means every audit
    /// runs via the local subprocess only.
    #[arg(long, env = "AUDITOR_TRUSTED_WALLETS", default_value = "")]
    pub auditor_trusted_wallets: String,
}

impl Config {
    /// Parses `opoi_addresses` into a trimmed, non-empty list — the input
    /// to `StakePool::new`.
    pub fn opoi_address_list(&self) -> Vec<String> {
        self.opoi_addresses
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// The first configured address — the one OPOI_COLLATERAL_TXID/_VOUT
    /// auto-bootstraps at startup, and the payout fallback below.
    pub fn primary_opoi_address(&self) -> &str {
        self.opoi_addresses.split(',').next().map(str::trim).unwrap_or("")
    }

    /// Address that should sign/receive miner payouts — payout_address if
    /// set, otherwise falls back to the bridge's primary OPoI stake address.
    pub fn effective_payout_address(&self) -> &str {
        if self.payout_address.is_empty() {
            self.primary_opoi_address()
        } else {
            &self.payout_address
        }
    }

    /// B3-lite is opt-in: no receipt secret configured means no receipts are
    /// signed/recorded/sampled at all (rather than silently signing with an
    /// empty key, which would be a receipt anyone could forge).
    pub fn b3lite_enabled(&self) -> bool {
        !self.b3lite_receipt_secret.is_empty()
    }

    /// Parses `auditor_trusted_wallets` into a trimmed, non-empty list —
    /// same shape `opoi_address_list` gives `opoi_addresses`.
    pub fn auditor_trusted_wallet_list(&self) -> Vec<String> {
        self.auditor_trusted_wallets
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    pub fn load() -> Self {
        // Load .env before clap reads the process environment, mirroring the
        // Node/back-pool convention — but .env is gitignored from commit 1,
        // never carries real secrets into version control.
        dotenvy::dotenv().ok();
        Config::parse()
    }
}
