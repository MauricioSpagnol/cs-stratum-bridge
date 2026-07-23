use std::sync::Arc;

use sqlx::PgPool;

use crate::config::Config;
use crate::opoi::{OpoiEngine, ShardEngine};

/// Shared state for the HTTP API. The stratum proxy/engine wiring lives
/// separately in main.rs (it needs `Arc<dyn OpoiHandler>`, not this struct).
#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub engine: Arc<OpoiEngine>,
    pub shard_engine: Arc<ShardEngine>,
    pub cfg: Arc<Config>,
}
