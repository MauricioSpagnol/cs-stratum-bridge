mod config;
mod db;
mod error;
mod http;
mod miner_registry;
mod opoi;
mod payout;
mod proxy;
mod rpc;
mod stake_pool;
mod state;

use std::sync::Arc;
use std::time::Duration;

use opoi::b3lite_audit::B3LiteAuditor;
use opoi::handler::{AuditHandler, ExpertHandler, OpoiHandler, ShardHandler, SpeculativeHandler};
use opoi::shard_engine::ModelSourceConfig;
use opoi::speculative_engine::DraftModelConfig;
use opoi::{ExpertDispatcher, OpoiEngine, ShardEngine, SpeculativeEngine};
use rpc::CsdRpcClient;
use stake_pool::StakePool;
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = Arc::new(config::Config::load());
    let stake_pool = Arc::new(StakePool::new(cfg.opoi_address_list()));
    tracing::info!(
        listen_addr = %cfg.listen_addr,
        upstream_pool_addr = %cfg.upstream_pool_addr,
        opoi_addresses = ?stake_pool.addresses(),
        payout_address = %cfg.effective_payout_address(),
        "cs-stratum-bridge starting"
    );

    let db_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(&cfg.database_url)
        .await?;
    sqlx::migrate!("./migrations").run(&db_pool).await?;
    tracing::info!("database connected and migrated");

    let csd = Arc::new(CsdRpcClient::new(cfg.csd_rpc_url.clone(), cfg.csd_rpc_user.clone(), cfg.csd_rpc_pass.clone()));

    // Fail fast and loud if csd isn't reachable — the bridge is useless
    // without it, better to crash at startup than silently serve a broken
    // pool.
    let height = csd.get_chain_height().await?;
    tracing::info!(height, "csd reachable");

    let registry = Arc::new(miner_registry::MinerRegistry::new());
    let engine = Arc::new(OpoiEngine::new(csd.clone(), db_pool.clone(), registry.clone(), stake_pool.clone()));

    // F15-L: only meaningfully active once BOTH a draft model is configured
    // for some target model (DRAFT_MODEL_SOURCES_JSON) AND a draft-capable
    // miner connects (F17-D) — see ShardEngine's `speculative` field doc and
    // SpeculativeEngine's module doc. Constructed unconditionally (cheap;
    // an empty DraftModelConfig just makes `is_eligible` always false, same
    // end state as `None`, but this way `SpeculativeHandler` always has a
    // real implementor for the proxy's session-level wiring below).
    let speculative_engine = Arc::new(SpeculativeEngine::new(csd.clone(), db_pool.clone(), registry.clone(), stake_pool.clone(), DraftModelConfig::from_env()));
    let b3lite_cfg = opoi::b3lite::B3LiteConfig::from_config(&cfg);
    if b3lite_cfg.is_none() {
        tracing::info!("B3LITE_RECEIPT_SECRET not set — B3-lite receipts/sampling/audit disabled");
    }
    // D2 (2026-07-26 session): constructed here (moved up from where it used
    // to live, right before the proxy wiring at the bottom of this
    // function) because `ShardEngine` now needs the SAME instance to fan
    // out an EXPERT-graph layer range via `dispatch_and_join` — see
    // `shard_engine.rs`'s module doc. Still also handed to the proxy below
    // as the process's `Arc<dyn ExpertHandler>`, unchanged.
    let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
    let shard_engine = Arc::new(ShardEngine::new(
        csd.clone(), db_pool.clone(), registry.clone(), stake_pool.clone(), ModelSourceConfig::from_env(), Some(speculative_engine.clone()),
        b3lite_cfg.clone(), expert_dispatcher.clone(),
    ));

    engine
        .ensure_stake(&cfg.opoi_collateral_txid, cfg.opoi_collateral_vout, &cfg.opoi_endpoint, &cfg.opoi_model_id)
        .await?;

    // Recovery pass MUST run before any of the normal loops start (see
    // opoi::engine::recover_on_startup doc comment).
    engine.recover_on_startup().await?;
    // ShardEngine (F15-H) has no restart-recovery pass yet — in-flight shard
    // pipelines are simply lost on restart and re-picked-up fresh from
    // PENDING on the next poll tick (see shard_engine.rs's module doc).

    // B3-lite: rebuild MinerRegistry's in-memory ban set from the durable
    // consequence table — otherwise a restart would silently un-eject every
    // wallet ejected before it (see MinerRegistry::ban's doc comment).
    match db::repo::list_ejected_wallets(&db_pool).await {
        Ok(wallets) => {
            for wallet in &wallets {
                registry.ban(wallet);
            }
            if !wallets.is_empty() {
                tracing::info!(count = wallets.len(), "B3-lite: restored ejected-wallet bans from durable consequence history");
            }
        }
        Err(e) => tracing::warn!(error = %e, "B3-lite: failed to restore ejected-wallet bans at startup"),
    }

    spawn_interval(cfg.poll_interval_ms, {
        let engine = engine.clone();
        move || {
            let engine = engine.clone();
            async move {
                if let Err(e) = engine.poll_and_assign_tick().await {
                    tracing::warn!(error = %e, "poll_and_assign_tick failed");
                }
            }
        }
    });

    spawn_interval(cfg.poll_interval_ms, {
        let shard_engine = shard_engine.clone();
        move || {
            let shard_engine = shard_engine.clone();
            async move {
                if let Err(e) = shard_engine.poll_and_start_tick().await {
                    tracing::warn!(error = %e, "shard_engine poll_and_start_tick failed");
                }
            }
        }
    });

    // F15-L: mirrors ShardEngine's own retry-stalled-pipelines behavior —
    // re-dispatches a speculative pipeline's current shard-relay or
    // draft-generate step if nothing is actually in flight for it right now
    // (first dispatch found no eligible miner, or the assigned one
    // disconnected mid-step).
    spawn_interval(cfg.poll_interval_ms, {
        let speculative_engine = speculative_engine.clone();
        move || {
            let speculative_engine = speculative_engine.clone();
            async move {
                speculative_engine.retry_stalled_tick().await;
            }
        }
    });

    // B3-lite (see opoi/b3lite_audit.rs): constructed unconditionally, same
    // reasoning as `speculative_engine` above — when B3-lite is disabled
    // (no B3LITE_RECEIPT_SECRET), no receipt ever gets `sampled = true`
    // (see ShardEngine::record_b3lite_receipt's guard), so `audit_tick`
    // simply finds nothing pending every tick. Also doubles as this
    // process's `Arc<dyn AuditHandler>` for the proxy's stratum-audit-
    // dispatch reply path below, regardless of whether the periodic tick
    // itself is spawned.
    let b3lite_auditor = Arc::new(B3LiteAuditor::new(
        db_pool.clone(),
        ModelSourceConfig::from_env(),
        registry.clone(),
        cfg.auditor_cs_miner_bin.clone(),
        std::path::PathBuf::from(&cfg.auditor_cache_dir),
        cfg.auditor_trusted_wallet_list(),
    ));
    if b3lite_cfg.is_some() {
        spawn_interval(cfg.b3lite_audit_poll_interval_ms, {
            let b3lite_auditor = b3lite_auditor.clone();
            move || {
                let b3lite_auditor = b3lite_auditor.clone();
                async move {
                    b3lite_auditor.audit_tick().await;
                }
            }
        });
    }

    spawn_interval(cfg.reveal_poll_interval_ms, {
        let engine = engine.clone();
        move || {
            let engine = engine.clone();
            async move {
                if let Err(e) = engine.poll_reveal_tick().await {
                    tracing::warn!(error = %e, "poll_reveal_tick failed");
                }
            }
        }
    });

    spawn_interval(cfg.renew_interval_ms, {
        let engine = engine.clone();
        move || {
            let engine = engine.clone();
            async move {
                engine.renew_tick().await;
            }
        }
    });

    spawn_interval(cfg.payout_interval_ms, {
        let csd = csd.clone();
        let db_pool = db_pool.clone();
        let payout_address = cfg.effective_payout_address().to_string();
        let min_payout_cs = cfg.min_payout_cs;
        let pool_fee_percent = cfg.pool_fee_percent;
        move || {
            let csd = csd.clone();
            let db_pool = db_pool.clone();
            let payout_address = payout_address.clone();
            async move {
                payout::payout_tick(csd, db_pool, payout_address, min_payout_cs, pool_fee_percent).await;
            }
        }
    });

    let app_state = AppState { db: db_pool.clone(), engine: engine.clone(), shard_engine: shard_engine.clone(), cfg: cfg.clone() };
    let http_router = http::router(app_state);
    let http_addr = cfg.http_listen_addr.clone();
    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(&http_addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(error = %e, addr = %http_addr, "failed to bind HTTP listener");
                return;
            }
        };
        tracing::info!(addr = %http_addr, "HTTP API listening");
        if let Err(e) = axum::serve(listener, http_router).await {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    let handler: Arc<dyn OpoiHandler> = engine.clone();
    let shard_handler: Arc<dyn ShardHandler> = shard_engine.clone();
    let speculative_handler: Arc<dyn SpeculativeHandler> = speculative_engine.clone();
    let audit_handler: Arc<dyn AuditHandler> = b3lite_auditor.clone();
    let expert_handler: Arc<dyn ExpertHandler> = expert_dispatcher.clone();
    let proxy_task = tokio::spawn(proxy::listener::run(
        cfg.listen_addr.clone(),
        cfg.upstream_pool_addr.clone(),
        registry.clone(),
        handler,
        shard_handler,
        speculative_handler,
        audit_handler,
        expert_handler,
    ));

    tokio::select! {
        res = proxy_task => {
            if let Err(e) = res {
                tracing::error!(error = %e, "stratum proxy listener task panicked");
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("received shutdown signal");
        }
    }

    Ok(())
}

/// Spawns a task that calls `make_fut()` on a fixed interval, forever. The
/// first tick fires after one full interval (not immediately) — every loop
/// here is a "keep doing this periodically" background job, not a
/// startup-critical action (those already ran synchronously above).
fn spawn_interval<F, Fut>(interval_ms: u64, mut make_fut: F)
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
        interval.tick().await; // first tick fires immediately; consume it so the loop below is "every interval, starting one interval from now"
        loop {
            interval.tick().await;
            make_fut().await;
        }
    });
}
