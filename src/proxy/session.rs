//! One downstream (cs-miner) <-> upstream (real pool) stratum session.
//!
//! `mining.*` traffic is relayed byte-for-byte (modulo the trailing
//! newline) in both directions. `opoi.submit_result` from the miner is
//! intercepted here and never forwarded upstream — it's handled entirely
//! through the injected `OpoiHandler`. `opoi.*` from upstream is dropped
//! defensively (it should never happen: OPoI lives in the bridge now).

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::error::AppError;
use crate::miner_registry::MinerRegistry;
use crate::opoi::handler::{AuditHandler, OpoiHandler, ShardHandler, SpeculativeHandler};
use crate::opoi::wire::{
    build_audit_result_ack, build_audit_result_error, build_shard_result_ack, build_shard_result_error,
    build_submit_result_ack, build_submit_result_error, OpoiAuditResult, OpoiDraftResult, OpoiKvRollbackAck, OpoiShardResult,
    OpoiSubmitResultParams,
};

/// Shared, mutable "who is this connection" state, handed off between the
/// down-reader (which parses `mining.authorize` requests and stashes a
/// *candidate* wallet) and the up-reader (which confirms it once the real
/// pool actually accepts the authorize, i.e. `id:2` + `result:true`).
#[derive(Default, Clone)]
struct AuthState {
    /// Parsed from the downstream `mining.authorize` request, before the
    /// upstream pool has confirmed it.
    pending: Arc<Mutex<Option<String>>>,
    /// Confirmed once the upstream pool's authorize response comes back
    /// accepted. This is the wallet OPoI submissions on this connection are
    /// attributed to.
    confirmed: Arc<Mutex<Option<String>>>,
}

pub async fn run_session(
    downstream: TcpStream,
    upstream_addr: String,
    registry: Arc<MinerRegistry>,
    handler: Arc<dyn OpoiHandler>,
    shard_handler: Arc<dyn ShardHandler>,
    speculative_handler: Arc<dyn SpeculativeHandler>,
    audit_handler: Arc<dyn AuditHandler>,
) -> anyhow::Result<()> {
    let peer = downstream.peer_addr().ok();

    let upstream = match TcpStream::connect(&upstream_addr).await {
        Ok(stream) => stream,
        Err(e) => {
            tracing::warn!(
                upstream_addr = %upstream_addr,
                peer = ?peer,
                error = %e,
                "failed to dial upstream pool; dropping downstream connection"
            );
            return Ok(());
        }
    };

    let (down_read, down_write) = tokio::io::split(downstream);
    let (up_read, up_write) = tokio::io::split(upstream);

    let mut down_lines = BufReader::new(down_read).lines();
    let mut up_lines = BufReader::new(up_read).lines();

    let (down_tx, down_rx) = mpsc::unbounded_channel::<String>();
    let (up_tx, up_rx) = mpsc::unbounded_channel::<String>();

    // The down-writer and up-writer are the ONLY tasks that ever write to
    // their respective sockets, spawned so they run independently of (and
    // can be torn down separately from) the two reader loops below.
    let down_writer = tokio::spawn(writer_task(down_write, down_rx));
    let up_writer = tokio::spawn(writer_task(up_write, up_rx));

    let auth = AuthState::default();

    tokio::select! {
        _ = down_reader_loop(&mut down_lines, down_tx.clone(), up_tx.clone(), registry.clone(), handler.clone(), shard_handler.clone(), speculative_handler.clone(), audit_handler.clone(), auth.clone()) => {
            tracing::debug!(peer = ?peer, "downstream reader ended session");
        }
        _ = up_reader_loop(&mut up_lines, down_tx.clone(), registry.clone(), auth.clone()) => {
            tracing::debug!(peer = ?peer, "upstream reader ended session");
        }
    }

    // Cleanup: whichever side ended first, tear the whole session down,
    // unregister from the MinerRegistry, and release any OPoI assignment
    // this connection was holding — otherwise a mid-flight disconnect would
    // strand that request_id forever (it's already marked assigned, so the
    // poller would never offer it to anyone else).
    // Scoped so the parking_lot::MutexGuard (which is !Send) is dropped
    // before the .await below — otherwise the whole session future becomes
    // non-Send and can't be tokio::spawn'ed from the listener.
    let confirmed_wallet = { auth.confirmed.lock().clone() };
    if let Some(wallet) = confirmed_wallet {
        registry.unregister(&wallet);
        handler.on_disconnect(&wallet).await;
        shard_handler.on_disconnect(&wallet).await;
        speculative_handler.on_disconnect(&wallet).await;
        audit_handler.on_disconnect(&wallet).await;
        tracing::info!(peer = ?peer, wallet = %wallet, "session ended; unregistered miner and released its pending assignments");
    }

    down_writer.abort();
    up_writer.abort();

    Ok(())
}

/// The single writer for one direction of a session. Loops on the channel
/// and writes whatever comes in verbatim, appending a newline only if the
/// line doesn't already end in one (raw relayed lines from `.lines()` never
/// do; pre-built OPoI response lines always do).
async fn writer_task(mut write_half: WriteHalf<TcpStream>, mut rx: UnboundedReceiver<String>) {
    while let Some(mut line) = rx.recv().await {
        if !line.ends_with('\n') {
            line.push('\n');
        }
        if let Err(e) = write_half.write_all(line.as_bytes()).await {
            tracing::debug!(error = %e, "write error; closing writer");
            break;
        }
    }
}

/// Reads frames from the downstream (miner) socket. `mining.*` (and
/// anything unrecognized/unparseable) is relayed upstream unchanged.
/// `opoi.submit_result` is intercepted and handled entirely locally via
/// `handler`, never forwarded. `mining.authorize` is additionally parsed to
/// stash a candidate wallet for the up-reader to confirm.
async fn down_reader_loop(
    lines: &mut Lines<BufReader<ReadHalf<TcpStream>>>,
    down_tx: UnboundedSender<String>,
    up_tx: UnboundedSender<String>,
    registry: Arc<MinerRegistry>,
    handler: Arc<dyn OpoiHandler>,
    shard_handler: Arc<dyn ShardHandler>,
    speculative_handler: Arc<dyn SpeculativeHandler>,
    audit_handler: Arc<dyn AuditHandler>,
    auth: AuthState,
) {
    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break, // EOF
            Err(e) => {
                tracing::debug!(error = %e, "downstream read error");
                break;
            }
        };

        let value: Option<Value> = serde_json::from_str(&line).ok();
        let method = value.as_ref().and_then(|v| v.get("method")).and_then(Value::as_str);

        match method {
            Some("opoi.submit_result") => {
                let id = value
                    .as_ref()
                    .and_then(|v| v.get("id"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let params_val = value
                    .as_ref()
                    .and_then(|v| v.get("params"))
                    .and_then(|p| p.get(0))
                    .cloned();

                let parsed: Option<OpoiSubmitResultParams> =
                    params_val.and_then(|p| serde_json::from_value(p).ok());

                let Some(params) = parsed else {
                    let _ = down_tx.send(build_submit_result_error(
                        id,
                        serde_json::json!([20, "malformed opoi.submit_result", null]),
                    ));
                    continue;
                };

                let wallet = auth.confirmed.lock().clone();
                let Some(wallet) = wallet else {
                    let _ = down_tx.send(build_submit_result_error(
                        id,
                        AppError::Unauthorized.to_stratum_error(),
                    ));
                    continue;
                };

                match handler.handle_submit_result(&wallet, params).await {
                    Ok(submission_id) => {
                        let _ = down_tx.send(build_submit_result_ack(id, submission_id));
                    }
                    Err(app_err) => {
                        let _ = down_tx.send(build_submit_result_error(id, app_err.to_stratum_error()));
                    }
                }
            }
            Some("opoi.shard_result") => {
                let id = value.as_ref().and_then(|v| v.get("id")).cloned().unwrap_or(Value::Null);
                let params_val = value.as_ref().and_then(|v| v.get("params")).and_then(|p| p.get(0)).cloned();

                let parsed: Option<OpoiShardResult> = params_val.and_then(|p| serde_json::from_value(p).ok());

                let Some(result) = parsed else {
                    let _ = down_tx.send(build_shard_result_error(id, serde_json::json!([20, "malformed opoi.shard_result", null])));
                    continue;
                };

                let wallet = auth.confirmed.lock().clone();
                let Some(wallet) = wallet else {
                    let _ = down_tx.send(build_shard_result_error(id, AppError::Unauthorized.to_stratum_error()));
                    continue;
                };

                match shard_handler.handle_shard_result(&wallet, result).await {
                    Ok(()) => {
                        let _ = down_tx.send(build_shard_result_ack(id));
                    }
                    Err(app_err) => {
                        let _ = down_tx.send(build_shard_result_error(id, app_err.to_stratum_error()));
                    }
                }
            }
            Some("opoi.audit_result") => {
                let id = value.as_ref().and_then(|v| v.get("id")).cloned().unwrap_or(Value::Null);
                let params_val = value.as_ref().and_then(|v| v.get("params")).and_then(|p| p.get(0)).cloned();

                let parsed: Option<OpoiAuditResult> = params_val.and_then(|p| serde_json::from_value(p).ok());

                let Some(result) = parsed else {
                    let _ = down_tx.send(build_audit_result_error(id, serde_json::json!([20, "malformed opoi.audit_result", null])));
                    continue;
                };

                let wallet = auth.confirmed.lock().clone();
                let Some(wallet) = wallet else {
                    let _ = down_tx.send(build_audit_result_error(id, AppError::Unauthorized.to_stratum_error()));
                    continue;
                };

                match audit_handler.handle_audit_result(&wallet, result).await {
                    Ok(()) => {
                        let _ = down_tx.send(build_audit_result_ack(id));
                    }
                    Err(app_err) => {
                        let _ = down_tx.send(build_audit_result_error(id, app_err.to_stratum_error()));
                    }
                }
            }
            Some("opoi.draft_result") => {
                // No reply sent back down for this one — see
                // `SpeculativeHandler`'s doc comment for why.
                let params_val = value.as_ref().and_then(|v| v.get("params")).and_then(|p| p.get(0)).cloned();
                let parsed: Option<OpoiDraftResult> = params_val.and_then(|p| serde_json::from_value(p).ok());
                let wallet = auth.confirmed.lock().clone();
                match (parsed, wallet) {
                    (Some(result), Some(wallet)) => speculative_handler.handle_draft_result(&wallet, result).await,
                    (Some(_), None) => tracing::warn!("opoi.draft_result from an unauthorized connection; dropping"),
                    (None, _) => tracing::warn!("malformed opoi.draft_result payload; dropping"),
                }
            }
            Some("opoi.kv_rollback_ack") => {
                let params_val = value.as_ref().and_then(|v| v.get("params")).and_then(|p| p.get(0)).cloned();
                let parsed: Option<OpoiKvRollbackAck> = params_val.and_then(|p| serde_json::from_value(p).ok());
                let wallet = auth.confirmed.lock().clone();
                match (parsed, wallet) {
                    (Some(ack), Some(wallet)) => speculative_handler.handle_kv_rollback_ack(&wallet, ack).await,
                    (Some(_), None) => tracing::warn!("opoi.kv_rollback_ack from an unauthorized connection; dropping"),
                    (None, _) => tracing::warn!("malformed opoi.kv_rollback_ack payload; dropping"),
                }
            }
            Some("opoi.capabilities") => {
                // F17-D: fire-and-forget, no reply expected (mirrors
                // mining.set_difficulty's shape going the other way) — see
                // miner_registry.rs's `mark_draft_capable`. Only meaningful
                // once the wallet is confirmed (right after authorize), same
                // ordering requirement as any other per-wallet state here.
                if let Some(wallet) = auth.confirmed.lock().clone() {
                    let caps: Vec<String> = value
                        .as_ref()
                        .and_then(|v| v.get("params"))
                        .and_then(|p| p.get(0))
                        .and_then(|p| p.get("caps"))
                        .and_then(|c| c.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    if caps.iter().any(|c| c == "draft") {
                        registry.mark_draft_capable(&wallet);
                        tracing::info!(wallet = %wallet, "miner registered `draft` capability (F17-D)");
                    }
                    if caps.iter().any(|c| c == "auditor") {
                        registry.mark_auditor_capable(&wallet);
                        tracing::info!(wallet = %wallet, "miner registered `auditor` capability (B3-lite)");
                    }
                } else {
                    tracing::debug!("opoi.capabilities from an unauthorized connection; dropping");
                }
            }
            Some("mining.authorize") => {
                if let Some(v) = &value {
                    if let Some(addr_raw) = v.get("params").and_then(|p| p.get(0)).and_then(Value::as_str) {
                        let stripped = addr_raw.strip_prefix("SOLO:").unwrap_or(addr_raw);
                        let wallet_only = stripped.split('.').next().unwrap_or(stripped).to_string();
                        *auth.pending.lock() = Some(wallet_only);
                    }
                }
                // Forward unchanged — the real pool still needs to see (and
                // authorize) this itself.
                let _ = up_tx.send(line);
            }
            _ => {
                // Everything else, including unparseable lines and any
                // other mining.* method, passes through untouched.
                let _ = up_tx.send(line);
            }
        }
    }
}

/// Reads frames from the upstream (real pool) socket and relays them down
/// to the miner unchanged, except:
///  - `opoi.*` from upstream is dropped defensively (shouldn't happen).
///  - The response to `mining.authorize` (id:2, result:true) is watched
///    for, to promote the down-reader's pending candidate wallet into a
///    confirmed one and register it in the MinerRegistry.
async fn up_reader_loop(
    lines: &mut Lines<BufReader<ReadHalf<TcpStream>>>,
    down_tx: UnboundedSender<String>,
    registry: Arc<MinerRegistry>,
    auth: AuthState,
) {
    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break, // EOF
            Err(e) => {
                tracing::debug!(error = %e, "upstream read error");
                break;
            }
        };

        let value: Option<Value> = serde_json::from_str(&line).ok();
        let method = value.as_ref().and_then(|v| v.get("method")).and_then(Value::as_str);

        if let Some(m) = method {
            if m.starts_with("opoi.") {
                tracing::warn!(
                    method = %m,
                    "unexpected opoi.* message from upstream pool; dropping (OPoI is owned by the bridge)"
                );
                continue;
            }
        }

        if let Some(v) = &value {
            let is_authorize_response = v.get("id").and_then(Value::as_i64) == Some(2)
                && v.get("result").and_then(Value::as_bool) == Some(true);

            if is_authorize_response {
                let candidate = auth.pending.lock().take();
                if let Some(wallet) = candidate {
                    *auth.confirmed.lock() = Some(wallet.clone());
                    registry.register(wallet.clone(), down_tx.clone());
                    tracing::info!(wallet = %wallet, "miner authorized upstream; registered");
                }
            }
        }

        let _ = down_tx.send(line);
    }
}
