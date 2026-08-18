use std::net::SocketAddr;

use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::db;
use crate::state::AppState;

pub async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    match sqlx::query("SELECT 1").fetch_one(&state.db).await {
        Ok(_) => (StatusCode::OK, "ok").into_response(),
        Err(e) => {
            tracing::error!(error = %e, "healthz: database check failed");
            (StatusCode::SERVICE_UNAVAILABLE, "database unreachable").into_response()
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    pub request_id: String,
    /// Plaintext prompt — hex-encoded here at the HTTP boundary before being
    /// handed to the engine, so everything downstream (cache, opoi.assign)
    /// only ever deals in the hex form the miner expects.
    pub prompt: String,
}

#[derive(Debug, Serialize)]
struct AcceptedResponse {
    accepted: bool,
}

/// Constant-time byte comparison — `==` on the raw API key would let a
/// network attacker measure per-byte comparison bailout timing to recover
/// the key character-by-character. Length is compared normally first (that
/// alone isn't secret-sensitive); if lengths match, every byte pair is
/// still visited regardless of where the first mismatch is.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

pub async fn submit_prompt(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<PromptRequest>,
) -> impl IntoResponse {
    // Checked before the API key itself: HTTP_LISTEN_ADDR defaults to
    // 0.0.0.0 (requester-facing, meant to be reachable from outside this
    // host), so this endpoint has no other defense against a client
    // hammering it — including to brute-force the key below.
    if !state.prompt_rate_limiter.check(peer.ip()) {
        return (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded, slow down").into_response();
    }

    let provided = headers.get("x-opoi-api-key").and_then(|v| v.to_str().ok());
    let authorized = matches!(provided, Some(p) if constant_time_eq(p.as_bytes(), state.cfg.opoi_requester_api_key.as_bytes()));
    if !authorized {
        return (StatusCode::UNAUTHORIZED, "missing or invalid x-opoi-api-key").into_response();
    }

    let prompt_hex = hex::encode(body.prompt.as_bytes());
    // Fed to both engines: at intake time it isn't known yet whether this
    // request will turn out to be shard-routed (ShardEngine) or whole-model
    // (OpoiEngine) — that's only decided once getmodelmanifest resolves.
    state.shard_engine.receive_prompt(&body.request_id, prompt_hex.clone());
    match state.engine.receive_prompt(&body.request_id, prompt_hex).await {
        Ok(true) => (StatusCode::OK, Json(AcceptedResponse { accepted: true })).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, "request_id not found on-chain").into_response(),
        Err(e) => {
            tracing::error!(error = %e, "submit_prompt failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}

pub async fn get_submission(State(state): State<AppState>, Path(request_id): Path<String>) -> impl IntoResponse {
    // Only reports the currently-active (non-FAILED) submission for this
    // request_id, if any — a request that failed and was never retried
    // simply reports 404 here; check server logs for the failure reason.
    match db::repo::find_active_by_request_id(&state.db, &request_id).await {
        Ok(Some(sub)) => Json(sub).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "no active submission for this request_id").into_response(),
        Err(e) => {
            tracing::error!(error = %e, "get_submission failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}

pub async fn list_pending(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.engine.pending_snapshot())
}

/// D2 live-topology follow-up (2026-07-26): "how many parts was this
/// request's model split into, and which wallet/GPU is running each one,
/// right now" — see `ShardEngine::topology_snapshot`'s doc comment for the
/// full rationale (this is the ONLY place this data exists at all; cs-miner
/// never talks to the chain daemon directly, and the daemon itself only
/// ever sees already-SUBMITTED on-chain results, never live in-flight
/// assignment). Unauthenticated, same as `get_submission`/`list_pending`
/// above — purely informational, read-only, no wallet/collateral action.
pub async fn get_topology(State(state): State<AppState>, Path(request_id): Path<String>) -> impl IntoResponse {
    match state.shard_engine.topology_snapshot(&request_id) {
        Some(snapshot) => Json(snapshot).into_response(),
        None => (StatusCode::NOT_FOUND, "no shard pipeline for this request_id right now").into_response(),
    }
}
