use axum::extract::{Path, State};
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

pub async fn submit_prompt(State(state): State<AppState>, headers: HeaderMap, Json(body): Json<PromptRequest>) -> impl IntoResponse {
    let provided = headers.get("x-opoi-api-key").and_then(|v| v.to_str().ok());
    if provided != Some(state.cfg.opoi_requester_api_key.as_str()) {
        return (StatusCode::UNAUTHORIZED, "missing or invalid x-opoi-api-key").into_response();
    }

    let prompt_hex = hex::encode(body.prompt.as_bytes());
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
