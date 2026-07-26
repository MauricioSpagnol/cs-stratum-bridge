//! Push-based reporting of live per-shard/per-expert topology to an
//! optionally-configured cs-admin-manager instance.
//!
//! This is push, not pull: `cs-stratum-bridge` runs on whatever network its
//! operator (a solo miner or a pool) chooses — usually with no inbound
//! reachability at all (behind NAT, no port forwarding, no reason to open
//! one). A central admin panel can never assume it can reach into that
//! network to pull this data. Reporting outbound instead works the same way
//! it already needs to work to talk to `csd`'s RPC or fetch GGUF sources:
//! this process dials out, nothing dials in. Entirely opt-in — see
//! `Config::admin_report_enabled`.

use std::sync::Arc;
use std::time::Duration;

use crate::opoi::ShardEngine;

/// Merges `pool_id` into a serialized `TopologySnapshot` — the receiving
/// cs-admin-manager keys stored reports by `(pool_id, request_id)` since,
/// unlike a single bridge's own local pipeline map, more than one operator
/// could in principle report the same `request_id` (independent pools
/// racing the same on-chain request — see multi-bridge-pool-architecture).
fn with_pool_id(mut snapshot_json: serde_json::Value, pool_id: &str) -> serde_json::Value {
    if let serde_json::Value::Object(ref mut map) = snapshot_json {
        map.insert("pool_id".to_string(), serde_json::Value::String(pool_id.to_string()));
    }
    snapshot_json
}

pub async fn report_tick(shard_engine: &Arc<ShardEngine>, http: &reqwest::Client, report_url: &str, api_key: &str, pool_id: &str) {
    for request_id in shard_engine.active_request_ids() {
        let Some(snapshot) = shard_engine.topology_snapshot(&request_id) else { continue };

        let body = match serde_json::to_value(&snapshot) {
            Ok(v) => with_pool_id(v, pool_id),
            Err(e) => {
                tracing::warn!(error = %e, request_id = %request_id, "failed to serialize topology snapshot for reporting");
                continue;
            }
        };

        let url = format!("{}/report", report_url.trim_end_matches('/'));
        let result = http
            .post(&url)
            .header("x-admin-report-key", api_key)
            .json(&body)
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => tracing::warn!(status = %resp.status(), request_id = %request_id, url = %url, "admin topology report rejected"),
            Err(e) => tracing::warn!(error = %e, request_id = %request_id, url = %url, "admin topology report failed to send"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_pool_id_adds_the_field_without_disturbing_the_rest() {
        let snapshot = serde_json::json!({ "request_id": "r1", "parts": [] });
        let merged = with_pool_id(snapshot, "tmSomePoolAddress");
        assert_eq!(merged["pool_id"], "tmSomePoolAddress");
        assert_eq!(merged["request_id"], "r1");
    }

    #[test]
    fn with_pool_id_overwrites_any_preexisting_pool_id_field() {
        let snapshot = serde_json::json!({ "pool_id": "stale", "request_id": "r1" });
        let merged = with_pool_id(snapshot, "fresh");
        assert_eq!(merged["pool_id"], "fresh");
    }
}
