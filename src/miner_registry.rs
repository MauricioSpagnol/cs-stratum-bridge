//! Tracks currently-connected, authorized downstream miners so the (later,
//! separately-built) OPoI engine can push `opoi.assign` notifications to a
//! specific miner and round-robin who gets assigned next.
//!
//! Nothing in this file depends on the proxy internals: it only ever sees a
//! wallet address and a channel to write raw stratum lines to that miner's
//! downstream socket.

use std::collections::{HashMap, HashSet};

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
    /// D2 (2026-07-26 session): wallets that sent `opoi.capabilities` with
    /// `"expert"` right after authorize, mapped to the VRAM (in MB) they
    /// reported alongside it — a MAP rather than a `HashSet` like
    /// `draft_capable`/`auditor_capable` because host selection for expert
    /// dispatch needs to compare actual VRAM amounts, not just presence
    /// (see `pick_expert_host`'s doc comment). Documented wire shape for
    /// cs-miner's side to reconcile against (mirrors the existing
    /// `opoi.capabilities` `{"caps": [...]}` shape): `{"caps": ["expert"],
    /// "expert_vram_mb": <u64>}` — the miner's own `hardware.rs::
    /// detect_gpu_vram_mb()` value, reported once right after authorize
    /// (same timing `draft`/`auditor` already use), NOT per-dispatch.
    expert_capable_vram: Mutex<HashMap<String, u64>>,
}

impl MinerRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(IndexMap::new()),
            draft_capable: Mutex::new(HashSet::new()),
            banned: Mutex::new(HashSet::new()),
            auditor_capable: Mutex::new(HashSet::new()),
            expert_capable_vram: Mutex::new(HashMap::new()),
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
        // Same reasoning for `expert` capability/VRAM — a disconnected
        // miner's reported VRAM is stale the moment its connection drops
        // (could reconnect with different hardware, or never at all).
        self.expert_capable_vram.lock().remove(wallet);
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

    /// D2: records that `wallet` announced the `expert` capability along
    /// with its VRAM in MB (`opoi.capabilities`, `{"caps": ["expert"],
    /// "expert_vram_mb": N}`) — mirrors `mark_auditor_capable`, but stores
    /// the reported value instead of only presence. Overwrites any
    /// previously-reported value for the same wallet (a miner re-announcing
    /// after e.g. a driver reset is expected to report current VRAM, not be
    /// stuck with a stale figure). No-op if `wallet` isn't currently
    /// registered, same guard `mark_auditor_capable`/`mark_draft_capable` use.
    pub fn mark_expert_capable(&self, wallet: &str, vram_mb: u64) {
        if self.inner.lock().contains_key(wallet) {
            self.expert_capable_vram.lock().insert(wallet.to_string(), vram_mb);
        }
    }

    /// Picks the BEST (highest-VRAM) connected, non-banned, not-already-used
    /// wallet with at least `min_vram_mb` of reported VRAM — same
    /// eligibility filters `pick_auditor_capable` applies (connected + not
    /// banned + not already claimed by this same dispatch), but unlike that
    /// method's `.find()` (first eligible match against an operator-curated
    /// trust allow-list), this does `.filter()` then `.max_by_key()` over
    /// every wallet that self-announced the `expert` capability.
    ///
    /// Deliberately no `trusted`-allowlist parameter (unlike
    /// `pick_auditor_capable`): an auditor's own honesty has no backstop
    /// other than the operator's trust in it, so only operator-designated
    /// wallets may ever be dispatched one. An expert-hosting miner's output
    /// is NOT trusted on say-so — it's independently checked after the fact
    /// by a replica auditor re-running the router-agreement/FFN-agreement
    /// sub-checks (see the D2 scope doc's "MODELO DE REDUNDÂNCIA/
    /// VERIFICAÇÃO: routing-trace-pinned" section) — so any connected,
    /// sufficiently-VRAM-equipped, non-banned wallet is a legitimate
    /// candidate; there is no separate operator allow-list to intersect
    /// with here.
    ///
    /// `min_vram_mb` is the caller's responsibility to size correctly for
    /// the specific expert being dispatched (see `pick_expert_hosts_top_n`'s
    /// doc comment on where that number should ultimately come from — not
    /// invented in this file). Returns `None` if no eligible wallet clears
    /// the VRAM bar right now.
    pub fn pick_expert_host(&self, min_vram_mb: u64, exclude_wallets: &[String]) -> Option<String> {
        self.pick_expert_hosts_top_n(min_vram_mb, exclude_wallets, 1).into_iter().next()
    }

    /// Same eligibility filter as `pick_expert_host`, but returns up to `n`
    /// distinct wallets ordered by VRAM descending (ties broken by wallet
    /// address, for determinism) instead of just the single best one — for
    /// future redundancy (dispatching several independent replicas of the
    /// same expert to different hosts, per the D2 scope doc). `pick_expert_host`
    /// is simply `n == 1`.
    ///
    /// TODO(D2): `min_vram_mb` should come from the actual weight-slice
    /// size of the specific expert being dispatched, read from the GGUF/
    /// manifest metadata (`getmodelmanifest`'s `expert_pom_roots` or
    /// similar — see `ModelManifest`'s doc comment in `rpc/types.rs`,
    /// which today only parses `arch_type`/`num_layers`/`backbone_pom_root`/
    /// `status` and explicitly ignores the rest). That metadata isn't
    /// threaded through to this layer yet, so every caller of this method
    /// must pass an explicit `min_vram_mb` of its own choosing for now —
    /// this method does not invent or default one.
    pub fn pick_expert_hosts_top_n(&self, min_vram_mb: u64, exclude_wallets: &[String], n: usize) -> Vec<String> {
        if n == 0 {
            return Vec::new();
        }
        let inner = self.inner.lock();
        let expert_capable_vram = self.expert_capable_vram.lock();
        let banned = self.banned.lock();

        let mut eligible: Vec<(String, u64)> = expert_capable_vram
            .iter()
            .filter(|(wallet, vram_mb)| {
                **vram_mb >= min_vram_mb
                    && inner.contains_key(wallet.as_str())
                    && !banned.contains(wallet.as_str())
                    && !exclude_wallets.iter().any(|excluded| excluded == *wallet)
            })
            .map(|(wallet, vram_mb)| (wallet.clone(), *vram_mb))
            .collect();

        // Highest VRAM first; ties broken by wallet address so the result
        // is deterministic across calls with an identical eligible set.
        eligible.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        eligible.into_iter().take(n).map(|(wallet, _)| wallet).collect()
    }
}

impl Default for MinerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod expert_host_tests {
    use super::*;

    fn registered(registry: &MinerRegistry, wallet: &str) {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register(wallet.to_string(), tx);
    }

    #[test]
    fn pick_expert_host_prefers_highest_vram_among_eligible() {
        let registry = MinerRegistry::new();
        registered(&registry, "small");
        registered(&registry, "big");
        registered(&registry, "medium");
        registry.mark_expert_capable("small", 4_000);
        registry.mark_expert_capable("big", 24_000);
        registry.mark_expert_capable("medium", 12_000);

        assert_eq!(registry.pick_expert_host(1_000, &[]), Some("big".to_string()));
    }

    #[test]
    fn pick_expert_host_filters_below_min_vram() {
        let registry = MinerRegistry::new();
        registered(&registry, "small");
        registry.mark_expert_capable("small", 4_000);

        assert_eq!(registry.pick_expert_host(8_000, &[]), None);
    }

    #[test]
    fn pick_expert_host_excludes_already_used_wallets() {
        let registry = MinerRegistry::new();
        registered(&registry, "big");
        registered(&registry, "medium");
        registry.mark_expert_capable("big", 24_000);
        registry.mark_expert_capable("medium", 12_000);

        assert_eq!(registry.pick_expert_host(1_000, &["big".to_string()]), Some("medium".to_string()));
    }

    #[test]
    fn pick_expert_host_excludes_banned_wallets() {
        let registry = MinerRegistry::new();
        registered(&registry, "big");
        registered(&registry, "medium");
        registry.mark_expert_capable("big", 24_000);
        registry.mark_expert_capable("medium", 12_000);
        registry.ban("big");

        assert_eq!(registry.pick_expert_host(1_000, &[]), Some("medium".to_string()));
    }

    #[test]
    fn pick_expert_host_ignores_disconnected_wallets() {
        let registry = MinerRegistry::new();
        registered(&registry, "big");
        registry.mark_expert_capable("big", 24_000);
        registry.unregister("big");

        assert_eq!(registry.pick_expert_host(1_000, &[]), None);
    }

    #[test]
    fn mark_expert_capable_is_noop_for_unregistered_wallet() {
        let registry = MinerRegistry::new();
        registry.mark_expert_capable("ghost", 99_000);
        assert_eq!(registry.pick_expert_host(1, &[]), None);
    }

    #[test]
    fn pick_expert_hosts_top_n_returns_multiple_ranked_by_vram() {
        let registry = MinerRegistry::new();
        registered(&registry, "small");
        registered(&registry, "big");
        registered(&registry, "medium");
        registry.mark_expert_capable("small", 4_000);
        registry.mark_expert_capable("big", 24_000);
        registry.mark_expert_capable("medium", 12_000);

        let top2 = registry.pick_expert_hosts_top_n(1_000, &[], 2);
        assert_eq!(top2, vec!["big".to_string(), "medium".to_string()]);
    }

    #[test]
    fn pick_expert_hosts_top_n_zero_returns_empty() {
        let registry = MinerRegistry::new();
        registered(&registry, "big");
        registry.mark_expert_capable("big", 24_000);
        assert_eq!(registry.pick_expert_hosts_top_n(1, &[], 0), Vec::<String>::new());
    }

    #[test]
    fn pick_expert_hosts_top_n_ties_break_on_wallet_address() {
        let registry = MinerRegistry::new();
        registered(&registry, "zzz");
        registered(&registry, "aaa");
        registry.mark_expert_capable("zzz", 8_000);
        registry.mark_expert_capable("aaa", 8_000);

        let top = registry.pick_expert_hosts_top_n(1, &[], 2);
        assert_eq!(top, vec!["aaa".to_string(), "zzz".to_string()]);
    }

    #[test]
    fn unregister_clears_expert_capable_vram() {
        let registry = MinerRegistry::new();
        registered(&registry, "big");
        registry.mark_expert_capable("big", 24_000);
        registry.unregister("big");
        registered(&registry, "big"); // reconnect, no VRAM re-announced yet
        assert_eq!(registry.pick_expert_host(1, &[]), None);
    }
}
