use std::net::IpAddr;
use std::time::{Duration, Instant};

use dashmap::DashMap;

/// Per-IP fixed-window rate limiter for `HTTP_LISTEN_ADDR`, which defaults
/// to `0.0.0.0` (requester-facing, meant to be reachable from outside this
/// host) — without this, `submit_prompt` had no defense against a client
/// hammering it, whether to brute-force `OPOI_REQUESTER_API_KEY` or just to
/// burn CPU/DB capacity. Deliberately simple (in-memory, single window, no
/// external crate) rather than a token-bucket/sliding-window library — this
/// only needs to blunt abuse, not meter traffic precisely.
pub struct RateLimiter {
    max_requests: u32,
    window: Duration,
    buckets: DashMap<IpAddr, (Instant, u32)>,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self { max_requests, window, buckets: DashMap::new() }
    }

    /// Returns `true` if this request is allowed, `false` if `ip` has
    /// exceeded `max_requests` within the current window.
    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let mut entry = self.buckets.entry(ip).or_insert((now, 0));
        if now.duration_since(entry.0) >= self.window {
            *entry = (now, 1);
            return true;
        }
        entry.1 += 1;
        entry.1 <= self.max_requests
    }
}
