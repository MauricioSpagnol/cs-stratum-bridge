//! D2: per-expert dispatch fan-out + join barrier — see "ESCOPO DE
//! IMPLEMENTAÇÃO DO D2" in `CS COIN OPoI MELHOR IMPLEMENTAÇÃO.txt` for the
//! full design rationale (routing-trace-pinned verification model, why
//! cs-stratum-bridge dispatches at all, effort estimate history).
//!
//! Generalizes the exact "assign work to a specific trusted wallet,
//! correlate the async response" mechanism `b3lite_audit.rs` built for
//! B3-lite's live-audit dispatch (`OpoiAuditAssign`/`OpoiAuditResult` +
//! `AuditHandler` + `pending: DashMap<request_id, oneshot::Sender<_>>`) to
//! the case where a SINGLE token's SINGLE MoE layer needs K of these
//! outstanding at once, to K different wallets, each on its own clock —
//! not reinvented, copied and keyed more finely.
//!
//! ## Why `request_id` alone is no longer enough
//!
//! The dense pipeline's `ShardEngine::assignments` (`DashMap<String,
//! ShardAssignment>`) only ever has ONE shard in flight per request_id at a
//! time — a dense pipeline is a strict A-then-B-then-C sequence. A B3-lite
//! audit dispatch is likewise one-at-a-time per request_id. MoE breaks that
//! invariant: for one token at one `(layer_start, layer_end)` range, the
//! router selects a top-k SET of experts, and this module dispatches each
//! one to a (probably different) wallet CONCURRENTLY, then waits for every
//! one of them before combining and advancing to the next layer. So the
//! natural assignment key becomes the 4-tuple `(request_id, layer_start,
//! layer_end, expert_id)` — see `ExpertAssignmentKey`.
//!
//! ## What this module does NOT do (scope boundary)
//!
//! This module dispatches and joins ALREADY-SELECTED experts — it is not
//! the router. Per the routing-trace-pinned verification model, deciding
//! WHICH experts to run for a given token/layer (and their combine weights)
//! is the primary executor's job (whoever runs the real router — a
//! forward-looking integration point, not built by this pass: today
//! `ShardEngine::poll_and_start_tick` still skips any model whose Model
//! Execution Graph contains an `EXPERT` shard entirely, see that module's
//! doc comment — cloudservice hasn't started emitting real per-expert MEG
//! nodes yet, that's a parallel, separate piece of D2 work). This module's
//! job starts at "dispatch these already-chosen experts, with these
//! already-decided combine weights, and collect their outputs" — the
//! reusable fan-out/join-barrier primitive that integration point will call
//! into once it exists.
//!
//! ## Combine
//!
//! `combine_weighted` decodes each expert's `data_hex` the exact same way
//! cs-miner's own `TensorPayload::to_hex`/`from_hex` do (`shard_compute.rs`
//! — 4-byte little-endian f32, row-major per `shape`), does a plain
//! elementwise router-weighted sum, and re-encodes. This is new: unlike the
//! dense pipeline (which only ever RELAYS `data_hex` between shard stages,
//! never interprets it — see `shard_engine.rs`'s use of `ShardOutputWire`),
//! MoE's combine step genuinely needs the bridge to do real tensor
//! arithmetic on the received activations.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

use crate::error::AppError;
use crate::miner_registry::MinerRegistry;
use crate::opoi::wire::{self, ExpertInputWire, ExpertOutputWire, OpoiExpertAssign, OpoiExpertResult};

/// Uniquely identifies ONE expert's dispatched FFN computation for one
/// token's one MoE layer within one request — see this module's doc
/// comment for why `request_id` alone (the dense pipeline's/B3-lite's key)
/// is no longer sufficient once a single token/layer can have K of these in
/// flight concurrently.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExpertAssignmentKey {
    pub request_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    /// D2 multi-layer follow-up (2026-07-26): the ABSOLUTE layer index
    /// this dispatch's `expert_id` FFN runs at — added so two DIFFERENT
    /// layers' fan-outs within the SAME `(layer_start, layer_end)` range
    /// (now genuinely possible once every layer in a range is walked and
    /// dispatched individually, not just the range's last one) never
    /// collide in `assignments`/`pending`. For the pre-existing
    /// single-layer-range case this is always `layer_start` (or
    /// equivalently `layer_end - 1`) — behavior for that case is
    /// unchanged.
    pub layer_idx: u32,
    pub expert_id: u32,
}

/// One expert already selected (by the real router — see module doc) to
/// run for a given fan-out group, with the combine-time weight the router
/// assigned it and the input activation it needs to run over. This
/// module's input, not its output — it never decides which experts appear
/// in this list or what weight they carry.
pub struct SelectedExpert {
    pub expert_id: u32,
    pub weight: f32,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Per-expert-dispatch timeout — independent per `(layer_start, layer_end,
/// expert_id)`, deliberately NOT one global timeout for the whole fan-out
/// group (see the D2 scope doc: "Model timeout/reassignment per-expert-
/// dispatch independently ... since different experts may be hosted on
/// machines with very different latency"). Shorter than B3-lite's
/// `STRATUM_AUDIT_TIMEOUT` (180s, a whole audit replay) — a single expert's
/// FFN over one token's activation is a much smaller unit of work.
const EXPERT_DISPATCH_TIMEOUT: Duration = Duration::from_secs(45);

/// A timed-out (or disconnected-mid-flight) expert dispatch is retried
/// against a different eligible host this many times before that expert
/// (and therefore the whole fan-out group — see `dispatch_and_join`'s doc
/// comment on why a missing term fails the whole combine) is given up on.
const MAX_RETRIES_PER_EXPERT: u32 = 2;

/// One expert's outcome inside a fan-out group — success carries the
/// dispatch's router weight alongside the raw result (needed by
/// `combine_weighted`), failure carries a diagnostic string.
type ExpertOutcome = Result<(f32, OpoiExpertResult), String>;

pub struct ExpertDispatcher {
    registry: Arc<MinerRegistry>,
    /// Outstanding dispatches: assignment key -> the wallet expected to
    /// answer — checked by `handle_expert_result` before accepting a reply.
    /// Mirrors `B3LiteAuditor::audit_assignments` exactly, just keyed by
    /// `ExpertAssignmentKey` instead of bare `request_id`.
    assignments: DashMap<ExpertAssignmentKey, String>,
    /// Outstanding dispatches: assignment key -> the one-shot sender
    /// `dispatch_one` is awaiting on. Removed (and the oneshot fires) the
    /// moment a matching `opoi.expert_result` arrives.
    pending: DashMap<ExpertAssignmentKey, oneshot::Sender<OpoiExpertResult>>,
}

impl ExpertDispatcher {
    pub fn new(registry: Arc<MinerRegistry>) -> Self {
        Self { registry, assignments: DashMap::new(), pending: DashMap::new() }
    }

    /// Dispatches every selected expert of one `(request_id, layer_start,
    /// layer_end)` fan-out group CONCURRENTLY, each to its own VRAM-eligible
    /// host (see `MinerRegistry::pick_expert_host`), waits on ALL of them —
    /// the join barrier — and combines their outputs as a router-weighted
    /// sum. Never returns a partial/best-effort result: the weighted-sum
    /// combine has no sound way to proceed with a missing term, so if ANY
    /// expert exhausts its independent retries, the WHOLE group errors out
    /// (the caller is expected to retry the entire token/layer step later,
    /// same "stalls here until a retry mechanism exists" acceptance the
    /// dense pipeline's own `handle_shard_result_inner` already documents
    /// for its analogous submit-failure case).
    ///
    /// Host selection for every expert in the group is done UP FRONT,
    /// sequentially, each excluding every wallet already claimed by an
    /// earlier expert in the SAME group (`used_wallets`) — this guarantees
    /// no two experts of one fan-out group are ever sent to the same
    /// wallet, which would silently serialize work that's supposed to be
    /// parallel and — worse — make one slow/dishonest wallet a single point
    /// of failure for two "independent" outputs at once. If ANY expert in
    /// the group has zero eligible host at dispatch time, the whole group
    /// is abandoned immediately (`Err`) without dispatching any of the
    /// others — cheaper to fail fast here than to tie up other hosts on
    /// work that can never combine.
    ///
    /// `min_vram_mb_for` maps an `expert_id` to the VRAM (MB) needed to
    /// host it — see `MinerRegistry::pick_expert_hosts_top_n`'s TODO on why
    /// that number isn't computed inside this module (GGUF/manifest expert
    /// size metadata isn't threaded through to this layer yet).
    ///
    /// `pre_excluded` — real self-deadlock found via live regtest
    /// validation (2026-07-26): `ShardEngine`'s downstream connection for
    /// each wallet is read by a single sequential loop
    /// (`proxy/session.rs::down_reader_loop`) that awaits each message's
    /// handler to completion before reading the connection's next line. If
    /// the SAME wallet that's currently the dense/primary sender of the
    /// `opoi.shard_result` this dispatch was triggered from is ALSO picked
    /// as one of this group's expert hosts, that wallet's own
    /// `opoi.expert_result` reply (needed to unblock THIS call, since it's
    /// awaited from inside that same `handle_shard_result` call chain)
    /// physically arrives on the very socket whose read loop is blocked
    /// waiting for this call to return — a genuine deadlock, only ever
    /// "resolved" by this expert's own per-attempt timeout eventually
    /// firing (confirmed live: exactly `EXPERT_DISPATCH_TIMEOUT`, not
    /// sooner). Callers pass the request's current primary wallet here so
    /// it's excluded from selection up front — same mechanism as
    /// `used_wallets` below, just seeded before the loop starts instead of
    /// growing during it.
    #[allow(clippy::too_many_arguments)]
    pub async fn dispatch_and_join(
        &self,
        request_id: &str,
        layer_start: u32,
        layer_end: u32,
        layer_idx: u32,
        token_index: u32,
        model_id: &str,
        model_gguf_url: &str,
        model_gguf_sha256: &str,
        experts: Vec<SelectedExpert>,
        min_vram_mb_for: impl Fn(u32) -> u64,
        pre_excluded: &[String],
    ) -> Result<ExpertOutputWire, String> {
        if experts.is_empty() {
            return Err("dispatch_and_join called with an empty expert list".to_string());
        }
        if layer_idx < layer_start || layer_idx >= layer_end {
            return Err(format!("dispatch_and_join called with layer_idx={layer_idx} outside its own range [{layer_start},{layer_end})"));
        }

        // Up-front, sequential host selection (see doc comment above) —
        // seeded with `pre_excluded` so the current dense/primary wallet
        // (see this method's doc comment on `pre_excluded`) can never be
        // picked as an expert host for its own request.
        let mut used_wallets: Vec<String> = pre_excluded.to_vec();
        let mut planned: Vec<(SelectedExpert, String)> = Vec::new();
        for sel in experts {
            let min_vram = min_vram_mb_for(sel.expert_id);
            let Some(wallet) = self.registry.pick_expert_host(min_vram, &used_wallets) else {
                return Err(format!(
                    "no eligible expert host for expert_id={} (min_vram_mb={min_vram}) at layer {layer_idx} (range [{layer_start},{layer_end})) — abandoning whole fan-out group",
                    sel.expert_id
                ));
            };
            used_wallets.push(wallet.clone());
            planned.push((sel, wallet));
        }

        let dispatches = planned.into_iter().map(|(sel, wallet)| {
            self.dispatch_one_with_retry(
                request_id, layer_start, layer_end, layer_idx, token_index, model_id, model_gguf_url, model_gguf_sha256, sel, wallet, used_wallets.clone(),
            )
        });

        let outcomes: Vec<ExpertOutcome> = futures::future::join_all(dispatches).await;

        let mut successes: Vec<(f32, OpoiExpertResult)> = Vec::with_capacity(outcomes.len());
        for outcome in outcomes {
            match outcome {
                Ok(pair) => successes.push(pair),
                Err(e) => return Err(format!("expert dispatch failed, whole fan-out group abandoned: {e}")),
            }
        }

        combine_weighted(&successes)
    }

    /// Dispatches ONE expert, retrying against a different eligible host
    /// (excluding every wallet already tried for THIS expert, and every
    /// wallet already claimed by ANOTHER expert in the same fan-out group)
    /// up to `MAX_RETRIES_PER_EXPERT` times on timeout/disconnect — the
    /// "independent per-expert timeout/reassignment" the D2 scope doc calls
    /// for, as opposed to one global timeout for the whole group.
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_one_with_retry(
        &self,
        request_id: &str,
        layer_start: u32,
        layer_end: u32,
        layer_idx: u32,
        token_index: u32,
        model_id: &str,
        model_gguf_url: &str,
        model_gguf_sha256: &str,
        sel: SelectedExpert,
        mut wallet: String,
        mut group_used_wallets: Vec<String>,
    ) -> ExpertOutcome {
        let mut tried_wallets: Vec<String> = Vec::new();
        let mut attempt = 0u32;

        loop {
            match self
                .dispatch_one(request_id, layer_start, layer_end, layer_idx, token_index, model_id, model_gguf_url, model_gguf_sha256, &sel, &wallet)
                .await
            {
                Ok(result) => return Ok((sel.weight, result)),
                Err(e) => {
                    tried_wallets.push(wallet.clone());
                    attempt += 1;
                    if attempt > MAX_RETRIES_PER_EXPERT {
                        return Err(format!(
                            "expert_id={} exhausted {MAX_RETRIES_PER_EXPERT} retries (last error against wallet {wallet}: {e})",
                            sel.expert_id
                        ));
                    }
                    // Exclude every wallet already tried for this expert AND
                    // every wallet already claimed by a sibling expert in
                    // this same group — a retry must never collide with
                    // either.
                    let mut exclude = group_used_wallets.clone();
                    exclude.extend(tried_wallets.iter().cloned());
                    let min_vram = 0; // re-picking after a failure relaxes nothing about eligibility already proven at plan time; caller-supplied floor was already satisfied once, VRAM doesn't change between retries of the same expert.
                    let Some(next_wallet) = self.registry.pick_expert_host(min_vram, &exclude) else {
                        return Err(format!("expert_id={} has no remaining eligible host to retry against after attempt {attempt} (last error: {e})", sel.expert_id));
                    };
                    tracing::warn!(request_id = %request_id, expert_id = sel.expert_id, layer_start, layer_end, old_wallet = %wallet, new_wallet = %next_wallet, attempt, error = %e, "D2 expert dispatch failed, retrying against a different host");
                    // Track the newly-claimed wallet in the group's used set
                    // too, so OTHER sibling retries (if any happen
                    // concurrently) don't also pick it.
                    group_used_wallets.push(next_wallet.clone());
                    wallet = next_wallet;
                }
            }
        }
    }

    /// Single dispatch-and-await, no retry — the primitive
    /// `dispatch_one_with_retry` loops on. Mirrors `B3LiteAuditor::
    /// dispatch_via_stratum` almost line for line.
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_one(
        &self,
        request_id: &str,
        layer_start: u32,
        layer_end: u32,
        layer_idx: u32,
        token_index: u32,
        model_id: &str,
        model_gguf_url: &str,
        model_gguf_sha256: &str,
        sel: &SelectedExpert,
        wallet: &str,
    ) -> Result<OpoiExpertResult, String> {
        let key = ExpertAssignmentKey { request_id: request_id.to_string(), layer_start, layer_end, layer_idx, expert_id: sel.expert_id };

        let tx = self.registry.get(wallet).ok_or_else(|| format!("wallet {wallet} not connected"))?;

        let data_hex = f32_to_hex(&sel.data);
        let input_hash = sha256_hex(&f32_to_le_bytes(&sel.data));

        let assign = OpoiExpertAssign {
            request_id: request_id.to_string(),
            model_id: model_id.to_string(),
            layer_start,
            layer_end,
            layer_idx,
            expert_id: sel.expert_id,
            token_index,
            input: ExpertInputWire::Tensor { shape: sel.shape.clone(), data_hex },
            input_hash,
            model_gguf_url: model_gguf_url.to_string(),
            model_gguf_sha256: model_gguf_sha256.to_string(),
        };

        let (result_tx, result_rx) = oneshot::channel();
        self.assignments.insert(key.clone(), wallet.to_string());
        self.pending.insert(key.clone(), result_tx);

        if tx.send(wire::build_expert_assign_line(&assign)).is_err() {
            self.assignments.remove(&key);
            self.pending.remove(&key);
            return Err(format!("wallet {wallet}'s downstream channel is closed"));
        }

        let outcome = tokio::time::timeout(EXPERT_DISPATCH_TIMEOUT, result_rx).await;

        // Always clean up both maps regardless of outcome — a late-arriving
        // `opoi.expert_result` after a timeout must not be accepted for
        // whatever dispatch happens to reuse this exact key next (same
        // "never trust a stale entry" discipline `B3LiteAuditor::
        // dispatch_via_stratum` already documents).
        self.assignments.remove(&key);
        self.pending.remove(&key);

        match outcome {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(format!("expert-result oneshot dropped for wallet {wallet}")),
            Err(_) => Err(format!("expert dispatch to wallet {wallet} timed out after {EXPERT_DISPATCH_TIMEOUT:?}")),
        }
    }
}

#[async_trait::async_trait]
impl crate::opoi::handler::ExpertHandler for ExpertDispatcher {
    async fn handle_expert_result(&self, wallet: &str, result: OpoiExpertResult) -> Result<(), AppError> {
        let key = ExpertAssignmentKey {
            request_id: result.request_id.clone(),
            layer_start: result.layer_start,
            layer_end: result.layer_end,
            layer_idx: result.layer_idx,
            expert_id: result.expert_id,
        };

        {
            let Some(assignment) = self.assignments.get(&key) else {
                return Err(AppError::UnknownRequest);
            };
            if assignment.value() != wallet {
                return Err(AppError::NotAssignedToCaller);
            }
        }

        let Some((_, sender)) = self.pending.remove(&key) else {
            // Already timed out (dispatch_one's own cleanup already removed
            // both entries) — a late reply, not an error to the caller, but
            // nothing to deliver it to either.
            return Ok(());
        };
        let _ = sender.send(result);
        Ok(())
    }

    async fn on_disconnect(&self, wallet: &str) {
        // Drop (fail fast, not "wait for timeout") any outstanding dispatch
        // whose expected wallet just disconnected — mirrors
        // `B3LiteAuditor::on_disconnect` exactly.
        let stale: Vec<ExpertAssignmentKey> = self.assignments.iter().filter(|e| e.value() == wallet).map(|e| e.key().clone()).collect();
        for key in stale {
            self.assignments.remove(&key);
            self.pending.remove(&key); // dropping the Sender is enough to unblock the waiter
        }
    }
}

fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

fn f32_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn f32_to_hex(data: &[f32]) -> String {
    hex::encode(f32_to_le_bytes(data))
}

/// Reverses `f32_to_hex`'s encoding — same 4-byte little-endian layout
/// cs-miner's own `TensorPayload::from_hex` uses (`shard_compute.rs`), so a
/// real miner's `opoi.expert_result` decodes correctly here. `pub(crate)`
/// (not private): `shard_engine.rs`'s D2 integration
/// (`handle_expert_group_result`) reuses this exact decode to turn a dense
/// shard's `ShardOutputWire::data_hex` into the `Vec<f32>` activation every
/// selected expert in a fan-out group is dispatched with as input.
pub(crate) fn hex_to_f32(hex_str: &str) -> Result<Vec<f32>, String> {
    let bytes = hex::decode(hex_str).map_err(|e| format!("output data_hex was not valid hex: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("output data_hex length {} is not a multiple of 4", bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Pure combine step: elementwise router-weighted sum of every expert's
/// output tensor for one fan-out group — extracted out of
/// `dispatch_and_join` so it's testable without any real dispatch/network
/// involved. Every output must share the SAME shape (an expert's FFN never
/// changes the hidden-state shape) — a mismatch is a hard error, not a
/// best-effort skip, since a silently-dropped term would silently corrupt
/// the combined activation fed into the next layer.
fn combine_weighted(outcomes: &[(f32, OpoiExpertResult)]) -> Result<ExpertOutputWire, String> {
    let Some((_, first)) = outcomes.first() else {
        return Err("combine_weighted called with no expert outputs".to_string());
    };
    let shape = first.output.shape.clone();
    let expected_len: usize = shape.iter().product();

    let mut acc = vec![0f32; expected_len];
    for (weight, result) in outcomes {
        if result.output.shape != shape {
            return Err(format!(
                "expert_id={} output shape {:?} does not match group shape {:?} — cannot combine",
                result.expert_id, result.output.shape, shape
            ));
        }
        let values = hex_to_f32(&result.output.data_hex)?;
        if values.len() != expected_len {
            return Err(format!(
                "expert_id={} output has {} floats, shape {:?} expects {}",
                result.expert_id,
                values.len(),
                shape,
                expected_len
            ));
        }
        for (a, v) in acc.iter_mut().zip(values.iter()) {
            *a += weight * v;
        }
    }

    Ok(ExpertOutputWire { shape, data_hex: f32_to_hex(&acc) })
}

#[cfg(test)]
mod combine_tests {
    use super::*;

    fn result(expert_id: u32, shape: Vec<usize>, data: &[f32]) -> OpoiExpertResult {
        OpoiExpertResult {
            request_id: "req-1".into(),
            layer_start: 0,
            layer_end: 4,
            layer_idx: 3,
            expert_id,
            token_index: 0,
            output_hash: String::new(),
            output: ExpertOutputWire { shape, data_hex: f32_to_hex(data) },
        }
    }

    #[test]
    fn combine_weighted_single_expert_full_weight_is_identity() {
        let out = combine_weighted(&[(1.0, result(0, vec![3], &[1.0, 2.0, 3.0]))]).expect("combine ok");
        assert_eq!(hex_to_f32(&out.data_hex).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(out.shape, vec![3]);
    }

    #[test]
    fn combine_weighted_two_experts_weighted_sum() {
        let out = combine_weighted(&[
            (0.5, result(0, vec![2], &[2.0, 4.0])),
            (0.5, result(1, vec![2], &[10.0, 20.0])),
        ])
        .expect("combine ok");
        assert_eq!(hex_to_f32(&out.data_hex).unwrap(), vec![6.0, 12.0]);
    }

    #[test]
    fn combine_weighted_three_experts_arbitrary_weights() {
        let out = combine_weighted(&[
            (0.2, result(0, vec![1], &[10.0])),
            (0.3, result(1, vec![1], &[10.0])),
            (0.5, result(2, vec![1], &[10.0])),
        ])
        .expect("combine ok");
        let values = hex_to_f32(&out.data_hex).unwrap();
        assert!((values[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn combine_weighted_rejects_mismatched_shapes() {
        let outcome = combine_weighted(&[
            (0.5, result(0, vec![2], &[1.0, 2.0])),
            (0.5, result(1, vec![3], &[1.0, 2.0, 3.0])),
        ]);
        assert!(outcome.is_err());
    }

    #[test]
    fn combine_weighted_empty_is_error() {
        assert!(combine_weighted(&[]).is_err());
    }

    #[test]
    fn f32_hex_round_trips() {
        let data = vec![0.0f32, -1.5, 3.14159, f32::MIN, f32::MAX];
        let hex_str = f32_to_hex(&data);
        let decoded = hex_to_f32(&hex_str).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn hex_to_f32_rejects_non_multiple_of_4() {
        assert!(hex_to_f32("aabbcc").is_err());
    }
}

#[cfg(test)]
mod dispatch_tests {
    //! Unit tests for the fan-out/join-barrier/retry logic itself, using a
    //! fake downstream channel exactly the way `b3lite_audit.rs`'s own
    //! `stratum_dispatch_tests` module does — no live network needed.
    use super::*;
    use crate::opoi::handler::ExpertHandler;

    fn expert(id: u32, weight: f32, value: f32) -> SelectedExpert {
        SelectedExpert { expert_id: id, weight, shape: vec![1], data: vec![value] }
    }

    #[tokio::test]
    async fn dispatch_and_join_combines_two_experts_dispatched_to_different_wallets() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx_a, mut rx_a) = tokio::sync::mpsc::unbounded_channel();
        let (tx_b, mut rx_b) = tokio::sync::mpsc::unbounded_channel();
        registry.register("wallet-a".to_string(), tx_a);
        registry.register("wallet-b".to_string(), tx_b);
        registry.mark_expert_capable("wallet-a", 24_000);
        registry.mark_expert_capable("wallet-b", 24_000);

        let dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let d = dispatcher.clone();
        let handle = tokio::spawn(async move {
            d.dispatch_and_join(
                "req-1", 0, 4, 3, 0, "model-x", "http://gguf", "gguf-hash",
                vec![expert(0, 0.5, 10.0), expert(1, 0.5, 20.0)],
                |_expert_id| 1_000,
                &[],
            )
            .await
        });

        // Both experts must get exactly one line, to two DIFFERENT wallets
        // (never the same one twice) — proves the "no wallet used twice in
        // one group" invariant `dispatch_and_join`'s doc comment describes.
        let line_a = rx_a.recv().await.expect("wallet-a got a line");
        let line_b = rx_b.recv().await.expect("wallet-b got a line");
        assert!(line_a.contains("opoi.expert_assign"));
        assert!(line_b.contains("opoi.expert_assign"));

        // Reply to both with their respective expert_id, echoing the value
        // sent (a trivial "identity" fake miner).
        for (expert_id, wallet, value) in [(0u32, "wallet-a", 10.0f32), (1u32, "wallet-b", 20.0f32)] {
            let result = OpoiExpertResult {
                request_id: "req-1".into(),
                layer_start: 0,
                layer_end: 4,
                layer_idx: 3,
                expert_id,
                token_index: 0,
                output_hash: String::new(),
                output: ExpertOutputWire { shape: vec![1], data_hex: f32_to_hex(&[value]) },
            };
            dispatcher.handle_expert_result(wallet, result).await.expect("accepted");
        }

        let combined = handle.await.expect("task join").expect("dispatch_and_join ok");
        let values = hex_to_f32(&combined.data_hex).unwrap();
        assert!((values[0] - 15.0).abs() < 1e-5); // 0.5*10 + 0.5*20
    }

    #[tokio::test]
    async fn dispatch_and_join_fails_whole_group_when_no_eligible_host_for_one_expert() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx_a, _rx_a) = tokio::sync::mpsc::unbounded_channel();
        registry.register("wallet-a".to_string(), tx_a);
        registry.mark_expert_capable("wallet-a", 24_000);

        let dispatcher = ExpertDispatcher::new(registry);
        // Two experts requested, only one eligible host exists at all —
        // the second can never be dispatched, so the whole group must fail
        // fast without ever sending a line for the first.
        let result = dispatcher
            .dispatch_and_join("req-2", 0, 4, 3, 0, "model-x", "http://gguf", "gguf-hash", vec![expert(0, 0.5, 1.0), expert(1, 0.5, 2.0)], |_| 1_000, &[])
            .await;
        assert!(result.is_err());
    }

    /// Real self-deadlock this session found via live regtest validation
    /// (see `dispatch_and_join`'s `pre_excluded` doc comment): a wallet
    /// that's also the request's current dense/primary sender must never
    /// be picked as an expert host for its own request — even when it's
    /// the ONLY otherwise-eligible host, `pre_excluded` must still win.
    #[tokio::test]
    async fn dispatch_and_join_never_picks_a_pre_excluded_wallet_even_as_the_only_eligible_host() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx_primary, _rx_primary) = tokio::sync::mpsc::unbounded_channel();
        registry.register("primary-wallet".to_string(), tx_primary);
        registry.mark_expert_capable("primary-wallet", 24_000);

        let dispatcher = ExpertDispatcher::new(registry);
        let result = dispatcher
            .dispatch_and_join(
                "req-4", 0, 4, 3, 0, "model-x", "http://gguf", "gguf-hash",
                vec![expert(0, 1.0, 1.0)],
                |_| 1_000,
                &["primary-wallet".to_string()],
            )
            .await;
        assert!(result.is_err(), "the only registered expert-capable wallet is pre_excluded, so this must fail, not deadlock against it");
    }

    #[tokio::test]
    async fn dispatch_and_join_rejects_empty_expert_list() {
        let registry = Arc::new(MinerRegistry::new());
        let dispatcher = ExpertDispatcher::new(registry);
        let result = dispatcher.dispatch_and_join("req-3", 0, 4, 3, 0, "model-x", "http://gguf", "gguf-hash", vec![], |_| 1_000, &[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn dispatch_and_join_rejects_layer_idx_outside_its_own_range() {
        let registry = Arc::new(MinerRegistry::new());
        let dispatcher = ExpertDispatcher::new(registry);
        // layer_idx=4 is NOT inside [0,4) — must be rejected up front,
        // before ever attempting to pick a host or dispatch anything.
        let result = dispatcher.dispatch_and_join("req-3b", 0, 4, 4, 0, "model-x", "http://gguf", "gguf-hash", vec![expert(0, 1.0, 1.0)], |_| 1_000, &[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn handle_expert_result_rejects_wrong_wallet() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("trusted-wallet".to_string(), tx);
        let dispatcher = ExpertDispatcher::new(registry);
        let key = ExpertAssignmentKey { request_id: "req-1".into(), layer_start: 0, layer_end: 4, layer_idx: 3, expert_id: 0 };
        dispatcher.assignments.insert(key.clone(), "trusted-wallet".to_string());
        dispatcher.pending.insert(key, oneshot::channel().0);

        let result = OpoiExpertResult {
            request_id: "req-1".into(), layer_start: 0, layer_end: 4, layer_idx: 3, expert_id: 0, token_index: 0,
            output_hash: String::new(), output: ExpertOutputWire { shape: vec![1], data_hex: f32_to_hex(&[1.0]) },
        };
        let outcome = dispatcher.handle_expert_result("some-other-wallet", result).await;
        assert!(matches!(outcome, Err(AppError::NotAssignedToCaller)));
    }

    #[tokio::test]
    async fn handle_expert_result_rejects_unknown_key() {
        let registry = Arc::new(MinerRegistry::new());
        let dispatcher = ExpertDispatcher::new(registry);
        let result = OpoiExpertResult {
            request_id: "never-dispatched".into(), layer_start: 0, layer_end: 4, layer_idx: 3, expert_id: 0, token_index: 0,
            output_hash: String::new(), output: ExpertOutputWire { shape: vec![1], data_hex: f32_to_hex(&[1.0]) },
        };
        let outcome = dispatcher.handle_expert_result("any-wallet", result).await;
        assert!(matches!(outcome, Err(AppError::UnknownRequest)));
    }

    #[tokio::test]
    async fn on_disconnect_unblocks_a_waiting_dispatch_immediately() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("trusted-wallet".to_string(), tx);
        let dispatcher = ExpertDispatcher::new(registry);

        let key = ExpertAssignmentKey { request_id: "req-1".into(), layer_start: 0, layer_end: 4, layer_idx: 3, expert_id: 0 };
        let (result_tx, result_rx) = oneshot::channel::<OpoiExpertResult>();
        dispatcher.assignments.insert(key.clone(), "trusted-wallet".to_string());
        dispatcher.pending.insert(key.clone(), result_tx);

        dispatcher.on_disconnect("trusted-wallet").await;

        assert!(result_rx.await.is_err());
        assert!(!dispatcher.assignments.contains_key(&key));
    }

    #[test]
    fn expert_assignment_key_distinguishes_different_experts_same_request() {
        let a = ExpertAssignmentKey { request_id: "r".into(), layer_start: 0, layer_end: 4, layer_idx: 3, expert_id: 0 };
        let b = ExpertAssignmentKey { request_id: "r".into(), layer_start: 0, layer_end: 4, layer_idx: 3, expert_id: 1 };
        assert_ne!(a, b);
    }

    #[test]
    fn expert_assignment_key_distinguishes_different_layer_ranges_same_expert() {
        let a = ExpertAssignmentKey { request_id: "r".into(), layer_start: 0, layer_end: 4, layer_idx: 3, expert_id: 0 };
        let b = ExpertAssignmentKey { request_id: "r".into(), layer_start: 4, layer_end: 8, layer_idx: 7, expert_id: 0 };
        assert_ne!(a, b);
    }

    /// D2 multi-layer follow-up (2026-07-26): the exact collision this
    /// field was added to prevent — two DIFFERENT absolute layers within
    /// the SAME `(layer_start, layer_end)` range, same expert_id, must be
    /// distinguishable.
    #[test]
    fn expert_assignment_key_distinguishes_different_layer_idx_same_range_and_expert() {
        let a = ExpertAssignmentKey { request_id: "r".into(), layer_start: 0, layer_end: 4, layer_idx: 1, expert_id: 0 };
        let b = ExpertAssignmentKey { request_id: "r".into(), layer_start: 0, layer_end: 4, layer_idx: 2, expert_id: 0 };
        assert_ne!(a, b);
    }
}
