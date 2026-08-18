pub mod handlers;
pub mod rate_limit;

use axum::routing::{get, post};
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(handlers::healthz))
        .route("/cscoin/opoi/prompt", post(handlers::submit_prompt))
        .route("/cscoin/opoi/submissions/:request_id", get(handlers::get_submission))
        .route("/cscoin/opoi/pending", get(handlers::list_pending))
        .route("/cscoin/opoi/topology/:request_id", get(handlers::get_topology))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
