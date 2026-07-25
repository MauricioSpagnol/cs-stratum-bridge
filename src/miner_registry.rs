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
    /// B3-lite (see `opoi/b3lite_audit.rs`'s "EJECTED" consequence): wallets
    /// excluded from future OPoI dispatch after repeated confirmed-fraud
    /// audits. Deliberately NOT removed from `inner` and NOT disconnected —
    /// PoW `mining.*` traffic is unaffected; only `pick_next`/
    /// `pick_draft_capable` skip a banned wallet, so it simply never
    /// receives another OPoI assignment. In-memory only (like
    /// `draft_capable`) — but `main.rs` rebuilds this set from the durable
    /// `b3lite_consequences` table (`db::repo::list_ejected_wallets`) once
    /// at startup, before any dispatch loop runs, so a restart doesn't
    /// silently un-eject anyone.
    banned: Mutex<HashSet<String>>,
    /// B3-lite (see `opoi/b3lite_audit.rs`): wallets that sent
    /// `opoi.capabilities` with `"auditor"` right after authorize — mirrors
    /// `draft_capable` exactly. Announcing this alone does NOT make a
    /// wallet eligible for a real audit dispatch — `pick_auditor_capable`
    /// also requires the wallet to be in the operator's own
    /// `AUDITOR_TRUSTED_WALLETS` allow-list (passed in by the caller, not
    /// stored here — see that method's doc).
    auditor_capable: Mutex<HashSet<String>>,
}

impl MinerRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(IndexMap::new()),
            draft_capable: Mutex::new(HashSet::new()),
            banned: Mutex::new(HashSet::new()),
            auditor_capable: Mutex::new(HashSet::new()),
        }
    }

    /// Excludes `wallet` from all future `pick_next`/`pick_draft_capable`
    /// selections — see `banned`'s doc comment for the exact scope (OPoI
    /// dispatch only, PoW mining unaffected).
    pub fn ban(&self, wallet: &str) {
        self.banned.lock().insert(wallet.to_string());
    }

    pub fn is_banned(&self, wallet: &str) -> bool {
        self.banned.lock().contains(wallet)
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
        // Same reasoning for `auditor` capability.
        self.auditor_capable.lock().remove(wallet);
    }

    pub fn get(&self, wallet: &str) -> Option<UnboundedSender<String>> {
        self.inner.lock().get(wallet).cloned()
    }

    /// Round-robin: returns the wallet after `last` in insertion order,
    /// wrapping around, skipping any B3-lite-banned wallet (see `banned`'s
    /// doc comment). If `last` is None, or `last` is no longer registered
    /// (miner disconnected), falls back to the FIRST currently-registered,
    /// non-banned wallet. Returns None if nothing is registered, or every
    /// registered wallet is banned.
    pub fn pick_next(&self, last: &Option<String>) -> Option<String> {
        let inner = self.inner.lock();
        if inner.is_empty() {
            return None;
        }
        let banned = self.banned.lock();

        let start_idx = match last.as_ref().and_then(|wallet| inner.get_index_of(wallet)) {
            Some(idx) => (idx + 1) % inner.len(),
            None => 0,
        };

        (0..inner.len())
            .map(|offset| (start_idx + offset) % inner.len())
            .find_map(|idx| inner.get_index(idx).filter(|(wallet, _)| !banned.contains(wallet.as_str())).map(|(wallet, _)| wallet.clone()))
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
        let banned = self.banned.lock();

        let start_idx = match last.as_ref().and_then(|wallet| inner.get_index_of(wallet)) {
            Some(idx) => (idx + 1) % inner.len(),
            None => 0,
        };

        (0..inner.len())
            .map(|offset| (start_idx + offset) % inner.len())
            .find_map(|idx| {
                inner
                    .get_index(idx)
                    .filter(|(wallet, _)| draft_capable.contains(wallet.as_str()) && !banned.contains(wallet.as_str()))
                    .map(|(wallet, _)| wallet.clone())
            })
    }

    /// B3-lite: mirrors `mark_draft_capable`, for the `auditor` capability.
    pub fn mark_auditor_capable(&self, wallet: &str) {
        if self.inner.lock().contains_key(wallet) {
            self.auditor_capable.lock().insert(wallet.to_string());
        }
    }

    /// Picks a connected wallet eligible for a real audit dispatch: must
    /// have announced the `auditor` capability, must be in the operator's
    /// `trusted` allow-list (see `Config::auditor_trusted_wallets` —
    /// announcing the capability alone is never sufficient, see
    /// `b3lite_audit.rs`'s module doc on why an arbitrary miner auditing a
    /// peer adds no security), must not be banned, and must not be
    /// `exclude_wallet` (the wallet actually being audited — auditing
    /// yourself proves nothing). No round-robin state here (unlike
    /// `pick_next`/`pick_draft_capable`): audit dispatch is rare enough
    /// (a small sampled fraction) that picking the first eligible match is
    /// fine. `None` if no eligible wallet is connected right now — callers
    /// must treat that as "fall back to the local subprocess auditor", not
    /// an error.
    pub fn pick_auditor_capable(&self, trusted: &[String], exclude_wallet: &str) -> Option<String> {
        let inner = self.inner.lock();
        let auditor_capable = self.auditor_capable.lock();
        let banned = self.banned.lock();

        trusted
            .iter()
            .find(|wallet| {
                wallet.as_str() != exclude_wallet
                    && inner.contains_key(wallet.as_str())
                    && auditor_capable.contains(wallet.as_str())
                    && !banned.contains(wallet.as_str())
            })
            .cloned()
    }
}

impl Default for MinerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
