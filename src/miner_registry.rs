//! Tracks currently-connected, authorized downstream miners so the (later,
//! separately-built) OPoI engine can push `opoi.assign` notifications to a
//! specific miner and round-robin who gets assigned next.
//!
//! Nothing in this file depends on the proxy internals: it only ever sees a
//! wallet address and a channel to write raw stratum lines to that miner's
//! downstream socket.

use indexmap::IndexMap;
use parking_lot::Mutex;
use tokio::sync::mpsc::UnboundedSender;

pub struct MinerRegistry {
    // Insertion-ordered map: wallet address -> a channel to push raw lines
    // to that miner's downstream socket.
    inner: Mutex<IndexMap<String, UnboundedSender<String>>>,
}

impl MinerRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(IndexMap::new()),
        }
    }

    pub fn register(&self, wallet: String, tx: UnboundedSender<String>) {
        self.inner.lock().insert(wallet, tx);
    }

    pub fn unregister(&self, wallet: &str) {
        // shift_remove (not swap_remove) preserves insertion order for the
        // remaining entries, which pick_next's fallback semantics rely on.
        self.inner.lock().shift_remove(wallet);
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
}

impl Default for MinerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
