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
use crate::opoi::handler::OpoiHandler;
use crate::opoi::wire::{build_submit_result_ack, build_submit_result_error, OpoiSubmitResultParams};

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
        _ = down_reader_loop(&mut down_lines, down_tx.clone(), up_tx.clone(), handler.clone(), auth.clone()) => {
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
    handler: Arc<dyn OpoiHandler>,
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
