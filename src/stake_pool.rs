//! Multi-identity OPoI stake pool — raises the odds of VRF eligibility.
//!
//! `nOPoIVrfThreshold`/`nOPoICoordinatorThreshold`/`nOPoIShardThreshold`
//! (cloudservice's `consensus/params.h`) gate every response-reveal /
//! coordinator-claim / shard-result submission behind ECVRF sortition,
//! seeded per `(address, request_id[, domain])`. The default threshold
//! grants ~3% eligibility per address on BOTH mainnet and testnet — only
//! `CRegTestParams` (`chainparams.cpp`) relaxes this, for testing only. With
//! a single custodied identity (the bridge's "pool mode", see `CS COIN OPoI
//! MELHOR IMPLEMENTAÇÃO.txt` Sprint 9), that ~3% already gates the
//! whole-model REVEAL today, and would compound per-shard once the shard
//! pipeline (F15-H) is restored (coordinator claim + one gate per shard).
//!
//! There is no read-only "am I eligible" RPC — `CheckVrfSortition` runs only
//! as a side effect of the real submit RPC, which fails with "VRF proof
//! generation failed" on ineligibility (confirmed in cloudservice's
//! `rpc/opoi.cpp`). So `try_each` really does call the real RPC per address,
//! in rotation, until one succeeds — there's no cheaper way to check first.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::AppError;

pub struct StakePool {
    addresses: Vec<String>,
    next: AtomicUsize,
}

impl StakePool {
    pub fn new(addresses: Vec<String>) -> Self {
        assert!(
            !addresses.is_empty(),
            "StakePool requires at least one OPoI address (check OPOI_ADDRESSES)"
        );
        Self { addresses, next: AtomicUsize::new(0) }
    }

    pub fn addresses(&self) -> &[String] {
        &self.addresses
    }

    /// The first configured address — used for startup stake
    /// auto-bootstrap and as the payout fallback (see `Config`).
    pub fn primary(&self) -> &str {
        &self.addresses[0]
    }

    /// Tries `f` against each pool address, starting from a rotating offset
    /// (so repeated calls don't always hit the same address first) and
    /// wrapping around the whole pool exactly once. Returns the first `Ok`
    /// along with which address produced it; returns the last `Err` if
    /// every address failed — callers should treat that as "not eligible
    /// this round, retry next tick," exactly like the pre-pool
    /// single-address behavior, never as a fatal error.
    pub async fn try_each<T, F, Fut>(&self, mut f: F) -> Result<(String, T), AppError>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = Result<T, AppError>>,
    {
        let len = self.addresses.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed) % len;
        let mut last_err = AppError::Rpc("StakePool: no addresses configured".to_string());
        for i in 0..len {
            let addr = self.addresses[(start + i) % len].clone();
            match f(addr.clone()).await {
                Ok(v) => return Ok((addr, v)),
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }
}
