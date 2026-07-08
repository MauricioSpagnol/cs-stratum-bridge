//! Tracks which connected miner (by wallet address) each pending OPoI
//! `request_id` was actually pushed an `opoi.assign` for — the piece that
//! was completely missing in the Node.js prototype this bridge replaces
//! (any authorized miner could answer any request_id it happened to know).

use std::time::Instant;

use dashmap::DashMap;

struct Assignment {
    miner_wallet: String,
    #[allow(dead_code)] // kept for future stuck-assignment diagnostics/metrics
    assigned_at: Instant,
}

#[derive(Default)]
pub struct AssignmentTracker(DashMap<String, Assignment>);

impl AssignmentTracker {
    pub fn new() -> Self {
        Self(DashMap::new())
    }

    /// Records that `request_id` was just pushed to `wallet`. Uses
    /// `entry().or_insert_with()` so this is atomic against a concurrent
    /// assignment attempt for the same request_id — returns `true` only if
    /// this call actually created the entry (i.e. it's safe to send the
    /// `opoi.assign` line); a `false` means someone else already claimed
    /// this request_id first and no assign should be sent.
    pub fn try_assign(&self, request_id: &str, wallet: &str) -> bool {
        let mut inserted = false;
        self.0.entry(request_id.to_string()).or_insert_with(|| {
            inserted = true;
            Assignment {
                miner_wallet: wallet.to_string(),
                assigned_at: Instant::now(),
            }
        });
        inserted
    }

    /// Returns the wallet a request_id is currently assigned to, if any.
    pub fn assignee(&self, request_id: &str) -> Option<String> {
        self.0.get(request_id).map(|a| a.miner_wallet.clone())
    }

    /// Removes an assignment once its submission has been durably recorded
    /// (the request is now "spoken for" and must never be reassigned).
    pub fn remove(&self, request_id: &str) {
        self.0.remove(request_id);
    }

    /// Called when a miner disconnects: releases every request_id it was
    /// holding an assignment for, without a submission, so the next
    /// assignment poll tick can hand them to a different connected miner —
    /// deliberately different from the back-pool prototype, which never
    /// re-offered an assignment once handed out, even after the assignee
    /// vanished.
    pub fn release_for_wallet(&self, wallet: &str) {
        self.0.retain(|_, a| a.miner_wallet != wallet);
    }
}
