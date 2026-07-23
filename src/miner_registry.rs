//! Tracks currently-connected, authorized downstream miners so the (later,
//! separately-built) OPoI engine can push `opoi.assign` notifications to a
//! specific miner and round-robin who gets assigned next.
//!
//! Nothing in this file depends on the proxy internals: it only ever sees a
//! wallet address and a channel to write raw stratum lines to that miner's
//! downstream socket.

use std::collections::HashSet;

use indexmap::IndexMap;
use parking_lot::Mutex;
use tokio::sync::mpsc::UnboundedSender;

pub struct MinerRegistry {
    // Insertion-ordered map: wallet address -> a channel to push raw lines
    // to that miner's downstream socket.
    inner: Mutex<IndexMap<String, UnboundedSender<String>>>,
    /// F17-D: wallets that sent `opoi.capabilities` with `"draft"` right
    /// after authorize (see `proxy/session.rs`'s interception of that
    /// method) — a SEPARATE set rather than a flag alongside `inner`'s
    /// entries so `unregister` (which drops the whole entry) and this can
    /// be reasoned about independently; cleared for a wallet on
    /// `unregister` regardless of whether it was ever marked, same as any
    /// other per-connection state.
    draft_capable: Mutex<HashSet<String>>,
}

impl MinerRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(IndexMap::new()),
            draft_capable: Mutex::new(HashSet::new()),
        }
    }

    pub fn register(&self, wallet: String, tx: UnboundedSender<String>) {
        self.inner.lock().insert(wallet, tx);
    }

    pub fn unregister(&self, wallet: &str) {
        // shift_remove (not swap_remove) preserves insertion order for the
        // remaining entries, which pick_next's fallback semantics rely on.
        self.inner.lock().shift_remove(wallet);
        // A disconnected miner is never `draft`-eligible again until it
        // reconnects and re-announces — stale entries here would otherwise
        // make pick_draft_capable hand out a wallet with no live connection.
        self.draft_capable.lock().remove(wallet);
    }

    pub fn get(&self, wallet: &str) -> Option<UnboundedSender<String>> {
        self.inner.lock().get(wallet).cloned()
    }

    /// Round-robin: returns the wallet after `last` in insertion order,
    /// wrapping around. If `last` is None, or `last` is no longer
    /// registered (miner disconnected), falls back to the FIRST
    /// currently-registered wallet. Returns None if nothing is registered.
    pub fn pick_next(&self, last: &Option<String>) -> Option<String> {
        let inner = self.inner.lock();
        if inner.is_empty() {
            return None;
        }

        let next_idx = match last.as_ref().and_then(|wallet| inner.get_index_of(wallet)) {
            Some(idx) => (idx + 1) % inner.len(),
            None => 0,
        };

        inner.get_index(next_idx).map(|(wallet, _)| wallet.clone())
    }

    /// F17-D: records that `wallet` announced the `draft` capability
    /// (`opoi.capabilities`, `{"caps": ["draft"]}`) — see
    /// `proxy/session.rs`'s interception of that method. A no-op (not an
    /// error) if `wallet` isn't currently registered at all — the caller
    /// (session.rs) already only calls this once the wallet is confirmed,
    /// but the check costs nothing and avoids ever marking a wallet that
    /// was never actually authorized.
    pub fn mark_draft_capable(&self, wallet: &str) {
        if self.inner.lock().contains_key(wallet) {
            self.draft_capable.lock().insert(wallet.to_string());
        }
    }

    /// Mirrors `pick_next`'s round-robin-ish fallback semantics, filtered to
    /// only wallets that registered the `draft` capability. `None` if no
    /// draft-capable miner is currently connected — callers must treat that
    /// as "fall back to non-speculative dispatch", not an error (see
    /// `speculative_engine.rs`'s integration point in `shard_engine.rs`).
    pub fn pick_draft_capable(&self, last: &Option<String>) -> Option<String> {
        let inner = self.inner.lock();
        let draft_capable = self.draft_capable.lock();
        if inner.is_empty() || draft_capable.is_empty() {
            return None;
        }

        let start_idx = match last.as_ref().and_then(|wallet| inner.get_index_of(wallet)) {
            Some(idx) => (idx + 1) % inner.len(),
            None => 0,
        };

        (0..inner.len())
            .map(|offset| (start_idx + offset) % inner.len())
            .find_map(|idx| inner.get_index(idx).filter(|(wallet, _)| draft_capable.contains(wallet.as_str())).map(|(wallet, _)| wallet.clone()))
    }
}

impl Default for MinerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
