//! F15-H (Sessão 3, Revisão v6): drives the DENSE shard pipeline for
//! requests routed to a multi-shard model, in parallel with `OpoiEngine`'s
//! existing whole-model flow. Not an extension of `OpoiEngine` — the
//! architecture is different enough (multi-stage dispatch, tensor relay
//! between stages, per-shard VRF submission) that keeping them separate
//! avoids a single struct doing two unrelated jobs.
//!
//! Job-routing split: `OpoiEngine::poll_and_assign_tick` now skips any
//! PENDING request whose `model` resolves to an ACTIVE `ModelManifest` —
//! those are this engine's job. A request with no manifest (still the
//! common case — most models aren't big enough to need sharding) is
//! whole-model and stays with `OpoiEngine`, unchanged.
//!
//! Tensor hand-off between pipeline stages happens THROUGH the bridge, not
//! miner-to-miner: `handle_shard_result` receives shard N's real output
//! tensor and immediately dispatches shard N+1 with it as input. This
//! avoids reopening the NAT/reachable-endpoint problem that motivated
//! `cs-stratum-bridge` existing in the first place — every miner already
//! has a live, authenticated stratum connection to the bridge; nothing new
//! needs to be reachable.
//!
//! D2 (2026-07-26 session): a model whose Model Execution Graph contains
//! `EXPERT` nodes is no longer skipped wholesale — cloudservice's
//! `BuildModelExecutionGraph` now emits real per-`(layer_range, expert_id)`
//! `EXPERT` nodes for MoE/HYBRID manifests (see "ESCOPO DE IMPLEMENTAÇÃO DO
//! D2" in `CS COIN OPoI MELHOR IMPLEMENTAÇÃO.txt`), laid out as: every
//! DENSE node for every layer range first (in range order), then every
//! EXPERT node grouped by layer range in that same order (`numExperts`
//! nodes per range). `poll_and_start_tick` now splits a fetched graph into
//! its DENSE nodes (`PipelineState::shards` — walked by `dispatch_current`/
//! `handle_shard_result_inner` exactly as before, byte-for-byte, for a pure
//! DENSE model) and its EXPERT nodes, grouped by the `(layer_start,
//! layer_end)` range of the DENSE shard they sit alongside
//! (`PipelineState::expert_groups`).
//!
//! When a dense shard's result comes back for a range that has a
//! registered EXPERT group, `handle_shard_result_inner` routes to
//! `handle_expert_group_result` instead of the plain dense-to-dense
//! handoff: that method fans the group out via
//! `expert_dispatch.rs::ExpertDispatcher::dispatch_and_join` (kept as its
//! own component, not folded into `PipelineState`/`ShardAssignment` below —
//! see that module's doc comment for why a K-way fan-out genuinely breaks
//! this file's "exactly one shard in flight per request" invariant) and
//! feeds the router-weighted-combined output forward as the next dense
//! shard's input, the same way a dense-to-dense handoff already works.
//!
//! **`router_selection` is now populated** by a real cs-miner build
//! (`shard_pool_client.rs::compute_moe_router_only`, see that crate's D2
//! session reports) — `OpoiShardResult::router_selection` (`wire.rs`)
//! carries the primary miner's real router choice, one entry per LAYER
//! within `(layer_start, layer_end)` (`Vec<Vec<RouterExpertChoice>>`, outer
//! index = layer offset within the range — mirrors `OpoiShardAssign`'s
//! sibling `pinned_expert_ids` convention, see `wire.rs`'s doc comment for
//! the full shape rationale). Still additive (`#[serde(default)]`), so an
//! older cs-miner build that never sets it still deserializes fine as
//! `None` — `handle_expert_group_result` logs a clear warning and stalls
//! the pipeline at that step whenever it's `None` (same "stalls here until
//! a retry mechanism exists" acceptance this file already documents for
//! submit failures below) — it never fabricates a selection.
//!
//! **Gap CLOSED (2026-07-26, multi-layer STATEFUL resume follow-up
//! session):** a dense-boundary range CAN legally span more than one layer
//! (cloudservice's `SplitLayerRanges` is a plain even split, nothing
//! requires `numDenseShards == numLayers` for a MoE manifest), and EVERY
//! layer in such a range is now genuinely, externally dispatched — not
//! just the range's last one. `PipelineState::moe_layer_walk` (see its own
//! doc comment) is the "genuinely new intermediate state" the PREVIOUS
//! session's note here called for: `Some((next_layer_idx, pinned_wallet))`
//! tracks "mid-range, at absolute layer `next_layer_idx`, pinned to the
//! wallet already holding this walk's open session" — a thin addition to
//! the existing `pos`/`current_input` shape, not a parallel state machine.
//! `dispatch_current` sets `OpoiShardAssign::moe_layer_step` to drive this
//! (`Some(layer_start)` on the FIRST dispatch for a range with a registered
//! EXPERT group, `Some(next_layer_idx)` on every resume — see that method's
//! doc); `handle_shard_result_inner` intercepts any INTERIOR-layer result
//! (`moe_layer_step` set, not yet the range's last layer) and routes it to
//! `handle_moe_interior_layer_result` — real dispatch/combine via
//! `ExpertDispatcher::dispatch_and_join` (now keyed by an explicit absolute
//! `layer_idx`, not just the range's bounds — see `expert_dispatch.rs`'s
//! `ExpertAssignmentKey` doc), but NEITHER submits on-chain NOR advances
//! `pos` — only the BOUNDARY layer's result (still handled by
//! `handle_expert_group_result`, essentially unchanged in substance) does
//! either of those, preserving the pre-existing "exactly one on-chain
//! submission per real dense shard_index" invariant untouched: the chain
//! never sees how many internal per-layer round trips happened before that
//! submission, only that it happened once, keyed the same way it always
//! was (see this crate's D2 session report, "on-chain submission-timing
//! choice", for the full reasoning this was checked against).
//!
//! A resume step MUST go back to the EXACT wallet pinned in
//! `moe_layer_walk` (never round-robin) — that wallet's own
//! `moe_range_session.rs` (cs-miner) is the only place the residual/KV
//! state needed to finish an interior layer's combine actually lives. If
//! that wallet disconnects mid-walk, `dispatch_current` drops the whole
//! pipeline rather than silently resuming against a different miner that
//! never computed the held-open state (see
//! `dispatch_current_drops_pipeline_when_pinned_wallet_for_a_mid_walk_resume_is_gone`).
//!
//! The OLD whole-range-in-one-shot path (`compute_moe_router_only`/
//! `route_only_over_range` on cs-miner's side, `handle_expert_group_result`
//! reading a MULTI-entry `router_selection` here) still exists and is still
//! exercised by `moe_multi_layer_range_dispatches_only_last_layer_selection`
//! below — it fires whenever a result's `moe_layer_step` is `None` (an
//! older cs-miner build, or any future caller that opts out of the new
//! protocol), and remains the documented fallback per this session's design
//! decision (see cs-miner's `route_only_over_range` doc comment for the
//! symmetric choice on that side).
//!
//! A second, narrower gap, unchanged from before: the LAST dense shard of a
//! MoE pipeline can also have a registered EXPERT group (every layer range
//! gets one when `numExperts > 0`), but there is no wire mechanism yet for
//! routing a combined MoE output into a final norm+lm_head/next-token
//! decode step — `handle_shard_result_inner` logs that case too and simply
//! falls back to the primary's own reported `next_token_id`, unchanged,
//! same as a non-expert-routed model.
//!
//! Scope note (Sessão 3, deliberate): pipeline state below is in-memory
//! only (`DashMap`), not persisted — a bridge restart loses in-flight shard
//! pipelines (they'll simply be re-picked-up from PENDING on the next poll
//! tick and restarted from token_index 0, same as a fresh request). Adding
//! restart-safe persistence for shard pipelines is future work, not a
//! correctness requirement for this first working version.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use sqlx::PgPool;

use crate::db;
use crate::miner_registry::MinerRegistry;
use crate::opoi::b3lite::{self, B3LiteConfig};
use crate::opoi::engine::do_commit;
use crate::opoi::expert_dispatch::{hex_to_f32, ExpertDispatcher, SelectedExpert};
use crate::opoi::expert_vram::ExpertVramEstimator;
use crate::opoi::speculative_engine::SpeculativeEngine;
use crate::opoi::wire::{build_shard_assign_line, OpoiShardAssign, ShardInputWire, ShardOutputWire};
use crate::rpc::types::{ModelGraph, ModelManifest, ShardDescriptor};
use crate::rpc::CsdRpcClient;
use crate::stake_pool::StakePool;

/// Where to download a model's GGUF/tokenizer from — not on-chain data (the
/// manifest only carries `backbone_pom_root`, the expected hash), so this is
/// local operator config. Loaded once from `MODEL_SOURCES_JSON` (env var: a
/// JSON object `{"MODEL_ID": {"gguf_url": "...", "tokenizer_url": "..."}}`).
/// A model with no entry here simply never gets shard-dispatched (treated
/// the same as "no manifest" by callers).
#[derive(Clone)]
pub struct ModelSourceConfig {
    sources: std::collections::HashMap<String, ModelSource>,
}

#[derive(Clone, serde::Deserialize)]
pub struct ModelSource {
    pub gguf_url: String,
    pub tokenizer_url: String,
    /// Not on-chain data (unlike the GGUF's `backbone_pom_root`) — the
    /// manifest carries no tokenizer hash, so this is operator-supplied
    /// config, same trust level as the URLs themselves. A wrong tokenizer
    /// only ever produces wrong (not unsafe) output, caught the same way
    /// any other divergence is: the protocol's N-of-M majority consensus.
    pub tokenizer_sha256: String,
}

impl ModelSourceConfig {
    pub fn from_env() -> Self {
        let raw = std::env::var("MODEL_SOURCES_JSON").unwrap_or_default();
        let sources = if raw.trim().is_empty() {
            std::collections::HashMap::new()
        } else {
            serde_json::from_str(&raw).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "MODEL_SOURCES_JSON is set but failed to parse; shard dispatch disabled for all models");
                std::collections::HashMap::new()
            })
        };
        Self { sources }
    }

    pub fn get(&self, model_id: &str) -> Option<&ModelSource> {
        self.sources.get(model_id)
    }
}

/// Per-request in-flight pipeline bookkeeping.
struct PipelineState {
    /// DENSE shards only, sorted by `shard_index` — validated at pipeline
    /// creation time to have no `EXPERT` entries (see `poll_and_dispatch_tick`).
    shards: Vec<ShardDescriptor>,
    manifest: ModelManifest,
    /// Carried through purely for the eventual `opoi_submissions` row (same
    /// informational-only field `OpoiEngine` fills from its own `pending`
    /// cache) — not used for any pipeline-control decision.
    prompt_hash: String,
    max_tokens: u32,
    /// Index into `shards` (NOT `shard_index` directly, though they
    /// coincide for a well-formed dense-only graph starting at 0).
    pos: usize,
    token_index: u32,
    /// What the currently-dispatched shard was given as input — kept so a
    /// re-dispatch (e.g. after a miner disconnects mid-shard) can resend
    /// the identical assignment instead of needing to reconstruct it.
    current_input: ShardInputWire,
    /// The ENTRY shard's original prompt, hex-encoded — captured once at
    /// pipeline creation and never overwritten (unlike `current_input`,
    /// which becomes `NextTokenId`/`Tensor` as generation advances). Needed
    /// at `finalize_pipeline` time to record a B3-lite receipt: a later
    /// audit replay must re-tokenize the SAME prompt this pipeline was
    /// actually served with (see `b3lite.rs`'s module doc).
    original_prompt_hex: String,
    /// The real decoded-token-id output of the LAST shard at every token
    /// step, in generation order — this IS the response, accumulated as it
    /// arrives (see `handle_shard_result_inner`). The bridge has no
    /// tokenizer of its own (only miners do, via `gguf_fetch.rs`/GGUF
    /// vocab), so unlike `OpoiEngine` (whose miners hand back decoded text
    /// directly as `response_hex`) there is no human-readable text
    /// available here to publish — see `finalize_pipeline`'s doc comment
    /// for how this is turned into a `response_hex`/`response_hash` pair
    /// that still satisfies the on-chain commit/reveal/publish
    /// hash-consistency requirement.
    generated_token_ids: Vec<u32>,
    /// Payout-attribution bookkeeping: how many shard-result steps each
    /// wallet actually delivered for this request. A dense pipeline
    /// round-robins EVERY shard dispatch across whichever miners are
    /// connected at that instant (see `dispatch_current`), so — unlike
    /// `OpoiEngine`, where exactly one assigned miner produces the entire
    /// response — a shard-routed request can genuinely involve several
    /// different downstream wallets. See `finalize_pipeline`'s doc comment
    /// for how this is resolved into a single payout attribution (the
    /// `opoi_submissions` schema only has room for one `miner_wallet` per
    /// on-chain response — a hard constraint tied to there being exactly
    /// one commit/reveal/publish per request_id, not a stylistic choice).
    contributions: BTreeMap<String, u32>,
    /// D2: `EXPERT` nodes from this request's Model Execution Graph,
    /// grouped by the `(layer_start, layer_end)` range of the DENSE shard
    /// they sit alongside — built once at pipeline-creation time (see
    /// `poll_and_start_tick`). Maps to the list of registered `expert_id`s
    /// for that range. Empty for a pure-DENSE model (no `EXPERT` nodes in
    /// the graph at all) — that emptiness is exactly what keeps the
    /// pure-DENSE path byte-for-byte unchanged (`expert_group_for` always
    /// returns `None`).
    expert_groups: HashMap<(u32, u32), Vec<u32>>,
    /// D2 multi-layer STATEFUL resume (2026-07-26 follow-up session — see
    /// cs-miner's `opoi::moe_range_session` module doc for the miner-side
    /// half of this protocol). `Some((next_layer_idx, pinned_wallet))`
    /// while the CURRENT shard (`shards[pos]`) is being walked layer-by-
    /// layer through a registered EXPERT range: `next_layer_idx` is the
    /// ABSOLUTE layer index the next (or currently in-flight)
    /// `opoi.shard_assign` for THIS SAME shard/pos should request via
    /// `moe_layer_step`, and `pinned_wallet` is the wallet already holding
    /// that range-walk's open session on its own side. A resume step MUST
    /// go back to this EXACT wallet, never round-robin to a different one
    /// — the session/residual state genuinely only exists on that one
    /// machine (see cs-miner's `moe_range_session.rs` doc for why). `None`
    /// whenever `pos`'s shard isn't (yet, or ever) being walked this way —
    /// a plain dense shard, or a fresh shard whose first dispatch hasn't
    /// happened yet. Reset to `None` whenever `pos` advances to a new
    /// shard (both the plain dense-to-dense handoff and the MoE boundary
    /// handoff clear it) so the next shard's dispatch decision is made
    /// fresh, not inherited from the shard that just finished.
    moe_layer_walk: Option<(u32, String)>,
}

impl PipelineState {
    /// Returns the registered `expert_id`s for shard `pos`'s layer range,
    /// if any were declared in this request's Model Execution Graph — the
    /// single decision point `handle_shard_result_inner` uses to tell a
    /// pure dense-to-dense handoff apart from one that must fan out through
    /// `ExpertDispatcher` first.
    fn expert_group_for(&self, pos: usize) -> Option<&Vec<u32>> {
        let shard = self.shards.get(pos)?;
        let key = (shard.layer_start.unwrap_or(0), shard.layer_end.unwrap_or(0));
        self.expert_groups.get(&key).filter(|ids| !ids.is_empty())
    }
}

/// Tracks which wallet the CURRENTLY in-flight shard step of each
/// request_id was assigned to — since a dense pipeline only ever has ONE
/// shard in flight per request at a time, keying by request_id alone (not
/// (request_id, shard_index)) is sufficient; `(shard_index, token_index)`
/// carried in `OpoiShardResult` guards against a stale/duplicate submission
/// answering a step that's no longer the current one.
#[derive(Default)]
struct ShardAssignment {
    wallet: String,
    shard_index: u32,
    token_index: u32,
    /// D2 multi-layer STATEFUL resume: echoes what `dispatch_current`
    /// actually asked for via `OpoiShardAssign::moe_layer_step` — validated
    /// against the miner's reply's OWN `moe_layer_step` before it's
    /// accepted (`handle_shard_result_inner`), so a stale/duplicate reply
    /// for a DIFFERENT layer step of the same shard/token can't be
    /// mistaken for the current one. `None` for a plain dense shard or the
    /// OLD whole-range-in-one-shot MoE path.
    moe_layer_step: Option<u32>,
}

pub struct ShardEngine {
    csd: Arc<CsdRpcClient>,
    db: PgPool,
    registry: Arc<MinerRegistry>,
    stake_pool: Arc<StakePool>,
    model_sources: ModelSourceConfig,
    pipelines: DashMap<String, PipelineState>,
    assignments: DashMap<String, ShardAssignment>,
    last_assigned: Mutex<Option<String>>,
    /// Own copy of request_id -> prompt_hex, fed by the same HTTP
    /// prompt-intake endpoint that feeds `OpoiEngine.pending` (see
    /// `receive_prompt` below and `http/handlers.rs`, which calls both).
    prompt_cache: DashMap<String, String>,
    /// F15-L: only `Some` when a draft model is configured for at least one
    /// model (`DRAFT_MODEL_SOURCES_JSON` non-empty — see main.rs). When
    /// present, `poll_and_start_tick` delegates a new request to it INSTEAD
    /// of starting a plain `PipelineState` here, whenever
    /// `SpeculativeEngine::is_eligible` says so (draft configured for this
    /// model AND a draft-capable miner is currently connected) — see this
    /// struct's module-level integration note and `speculative_engine.rs`'s
    /// module doc for the full design. `opoi.shard_result` for a
    /// speculatively-dispatched request_id is likewise forwarded to it
    /// (see `handle_shard_result_inner`) instead of processed here.
    speculative: Option<Arc<SpeculativeEngine>>,
    /// `None` when B3-lite is disabled (`Config::b3lite_enabled` false) —
    /// `finalize_pipeline` records no receipt at all in that case, rather
    /// than recording an unsigned one (see `B3LiteConfig`'s doc).
    b3lite: Option<B3LiteConfig>,
    /// D2: the SAME `ExpertDispatcher` instance main.rs also hands to the
    /// proxy as the process's `Arc<dyn ExpertHandler>` — sharing the
    /// instance (not constructing a second one) is what lets a downstream
    /// miner's `opoi.expert_result` reply actually reach the
    /// `dispatch_and_join` call this engine makes in
    /// `handle_expert_group_result` below.
    expert_dispatcher: Arc<ExpertDispatcher>,
    /// D2 follow-up: real per-expert VRAM sizing for `dispatch_and_join`'s
    /// `min_vram_mb_for` callback — see `expert_vram.rs`'s module doc for
    /// the full mechanism (GGUF-header-only HTTP read, no shared state with
    /// anything else, so this is simply self-constructed in `new` rather
    /// than threaded through every call site the way `expert_dispatcher`
    /// must be).
    expert_vram: ExpertVramEstimator,
}

impl ShardEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        csd: Arc<CsdRpcClient>, db: PgPool, registry: Arc<MinerRegistry>, stake_pool: Arc<StakePool>, model_sources: ModelSourceConfig,
        speculative: Option<Arc<SpeculativeEngine>>, b3lite: Option<B3LiteConfig>, expert_dispatcher: Arc<ExpertDispatcher>,
    ) -> Self {
        Self {
            csd,
            db,
            registry,
            stake_pool,
            model_sources,
            pipelines: DashMap::new(),
            assignments: DashMap::new(),
            last_assigned: Mutex::new(None),
            prompt_cache: DashMap::new(),
            speculative,
            b3lite,
            expert_dispatcher,
            expert_vram: ExpertVramEstimator::new(),
        }
    }

    /// Stashes a plaintext prompt for a request_id — called from the same
    /// HTTP handler that feeds `OpoiEngine::receive_prompt`, since at intake
    /// time it isn't known yet whether this request will turn out to be
    /// shard-routed or whole-model.
    pub fn receive_prompt(&self, request_id: &str, prompt_hex: String) {
        self.prompt_cache.insert(request_id.to_string(), prompt_hex);
    }

    /// Returns `true` if `model_id` resolves to an ACTIVE manifest — the
    /// exact check `OpoiEngine::poll_and_assign_tick` uses to decide whether
    /// a PENDING request belongs to this engine instead.
    pub async fn is_shard_routed(&self, model_id: &str) -> bool {
        matches!(self.csd.get_model_manifest(model_id).await, Ok(m) if m.status == "ACTIVE")
    }

    /// One tick: for every PENDING request whose model has an ACTIVE
    /// manifest and no pipeline yet, try to start one (claim coordinator +
    /// resolve the MEG + dispatch shard 0). Requests already mid-pipeline
    /// are advanced entirely by `handle_shard_result` — EXCEPT a pipeline
    /// whose current shard has no live assignment (first dispatch found no
    /// miner connected, or the assigned miner disconnected mid-shard): those
    /// are retried here too, since nothing else ever re-triggers dispatch
    /// for them otherwise (`on_disconnect` only clears the stale
    /// assignment, it doesn't redispatch — this tick is what does).
    pub async fn poll_and_start_tick(&self) -> anyhow::Result<()> {
        self.retry_stalled_pipelines().await;

        for req in self.csd.list_pending_requests().await? {
            if self.pipelines.contains_key(&req.request_id) {
                continue;
            }
            let Ok(manifest) = self.csd.get_model_manifest(&req.model).await else { continue };
            if manifest.status != "ACTIVE" {
                continue;
            }
            let Some(prompt_hex) = self.prompt_cache.get(&req.request_id).map(|e| e.clone()) else {
                continue; // no prompt cached for this request yet — same wait condition OpoiEngine has
            };

            let Ok(graph) = self.csd.get_model_graph(&req.model).await else {
                tracing::warn!(model = %req.model, "ACTIVE manifest but getmodelgraph failed; skipping this tick");
                continue;
            };
            let Some(source) = self.model_sources.get(&req.model) else {
                tracing::warn!(model = %req.model, "no MODEL_SOURCES_JSON entry — cannot resolve a download URL, skipping");
                continue;
            };

            // D2: split the graph into its DENSE nodes (walked exactly as
            // before by dispatch_current/handle_shard_result_inner) and its
            // EXPERT nodes, grouped by the (layer_start, layer_end) range of
            // the DENSE shard they sit alongside — see
            // `split_model_execution_graph`'s doc comment. A pure-DENSE
            // model has no EXPERT entries at all, so `expert_groups` ends up
            // empty and this whole pipeline behaves byte-for-byte as before.
            let (shards, expert_groups) = split_model_execution_graph(&graph.shards);
            if shards.is_empty() {
                tracing::warn!(model = %req.model, request_id = %req.request_id, "model graph has no DENSE shard nodes at all — cannot start a pipeline, skipping");
                continue;
            }

            // Self-claim coordinator across the whole pool before committing
            // to this request — if no address is eligible this round, leave
            // it PENDING and retry next tick (same "not eligible yet" retry
            // pattern as everything else VRF-gated in this protocol).
            let claim = self.stake_pool.try_each(|addr| {
                let csd = self.csd.clone();
                let request_id = req.request_id.clone();
                async move { csd.claim_coordinator(&request_id, &addr).await }
            }).await;
            if let Err(e) = claim {
                tracing::debug!(request_id = %req.request_id, error = %e, "no pool address eligible to coordinate this request yet, will retry");
                continue;
            }

            let source = source.clone();

            // F15-L: delegate to the speculative pipeline instead of
            // starting our own plain one, iff a draft model is configured
            // for this model AND a draft-capable miner is currently
            // connected — falls back to the plain per-token path below
            // unconditionally otherwise (including whenever `speculative`
            // itself is `None`, i.e. no draft model configured anywhere).
            if let Some(spec) = &self.speculative {
                if spec.is_eligible(&manifest.model_id) {
                    tracing::info!(request_id = %req.request_id, model = %manifest.model_id, "dispatching via SpeculativeEngine (F15-L)");
                    spec.start_pipeline(req.request_id.clone(), manifest, shards, req.max_tokens, prompt_hex.clone(), req.prompt_hash.clone(), source).await;
                    self.prompt_cache.remove(&req.request_id);
                    continue;
                }
            }

            self.pipelines.insert(
                req.request_id.clone(),
                PipelineState {
                    shards,
                    manifest,
                    prompt_hash: req.prompt_hash.clone(),
                    max_tokens: req.max_tokens,
                    pos: 0,
                    token_index: 0,
                    current_input: ShardInputWire::Prompt { prompt_hex: prompt_hex.clone() },
                    original_prompt_hex: prompt_hex.clone(),
                    generated_token_ids: Vec::new(),
                    contributions: BTreeMap::new(),
                    expert_groups,
                    moe_layer_walk: None,
                },
            );
            self.prompt_cache.remove(&req.request_id);
            self.dispatch_current(&req.request_id, &source).await;
        }
        Ok(())
    }

    /// Redispatches the current shard/token step of every pipeline that
    /// exists but has no live assignment right now — see
    /// `poll_and_start_tick`'s doc comment for why this is needed at all.
    /// Collects request_ids first (not holding any DashMap iterator guard
    /// across the subsequent `.await`s in `dispatch_current`).
    async fn retry_stalled_pipelines(&self) {
        let stalled: Vec<String> = self
            .pipelines
            .iter()
            .filter(|e| !self.assignments.contains_key(e.key()))
            .map(|e| e.key().clone())
            .collect();

        for request_id in stalled {
            let model_id = { self.pipelines.get(&request_id).map(|p| p.manifest.model_id.clone()) };
            let Some(model_id) = model_id else { continue };
            let Some(source) = self.model_sources.get(&model_id).cloned() else { continue };
            self.dispatch_current(&request_id, &source).await;
        }
    }

    /// D2 multi-layer STATEFUL resume (2026-07-26 follow-up session): now
    /// makes TWO decisions instead of one:
    ///   1. Which WALLET to dispatch to. Ordinarily this always
    ///      round-robins (`MinerRegistry::pick_next`) — but a RESUME step
    ///      of a per-layer walk (`pipeline.moe_layer_walk` already
    ///      `Some`) MUST go back to the EXACT wallet already holding that
    ///      walk's open session (see `PipelineState::moe_layer_walk`'s doc
    ///      for why — the residual/KV state genuinely only exists there).
    ///      If that pinned wallet is no longer connected, the whole
    ///      pipeline is dropped rather than silently resuming against a
    ///      different machine that never computed the held-open state.
    ///   2. Whether to set `moe_layer_step` at all. ANY dense shard with a
    ///      registered EXPERT group (any range length — see this file's
    ///      module doc for why length no longer matters) now uses the new
    ///      per-layer walk protocol UNCONDITIONALLY going forward: the
    ///      FIRST dispatch for such a shard pins a wallet and sets
    ///      `moe_layer_walk = Some((layer_start, wallet))`; every
    ///      subsequent call for the SAME shard/pos (until it advances)
    ///      just re-reads that pinned state. A plain dense shard (no
    ///      EXPERT group) is completely unaffected — `moe_layer_step`
    ///      stays `None`, byte-for-byte the same dispatch as before this
    ///      session.
    async fn dispatch_current(&self, request_id: &str, source: &ModelSource) {
        struct Snapshot {
            shard_index: u32,
            layer_start: u32,
            layer_end: u32,
            total_layers: u32,
            token_index: u32,
            max_tokens: u32,
            model_id: String,
            gguf_sha256: String,
            current_input: ShardInputWire,
            is_entry: bool,
            needs_walk: bool,
            moe_layer_walk: Option<(u32, String)>,
        }
        let Some(snap) = self.pipelines.get(request_id).map(|pipeline| {
            let shard = &pipeline.shards[pipeline.pos];
            Snapshot {
                shard_index: shard.shard_index,
                layer_start: shard.layer_start.unwrap_or(0),
                layer_end: shard.layer_end.unwrap_or(0),
                total_layers: pipeline.manifest.num_layers,
                token_index: pipeline.token_index,
                max_tokens: pipeline.max_tokens,
                model_id: pipeline.manifest.model_id.clone(),
                gguf_sha256: pipeline.manifest.backbone_pom_root.clone(),
                current_input: pipeline.current_input.clone(),
                is_entry: pipeline.pos == 0,
                needs_walk: pipeline.expert_group_for(pipeline.pos).is_some(),
                moe_layer_walk: pipeline.moe_layer_walk.clone(),
            }
        }) else {
            return;
        };

        let (wallet, moe_layer_step) = match snap.moe_layer_walk {
            Some((layer_idx, pinned_wallet)) => {
                if self.registry.get(&pinned_wallet).is_none() {
                    tracing::error!(
                        request_id = %request_id, wallet = %pinned_wallet, layer_idx,
                        "D2 multi-layer walk: the wallet holding this range-walk's open session is no longer connected — \
                         its residual/KV state cannot be recovered from a different miner; dropping the pipeline rather \
                         than silently resuming against a machine that never computed it"
                    );
                    self.pipelines.remove(request_id);
                    self.assignments.remove(request_id);
                    return;
                }
                (pinned_wallet, Some(layer_idx))
            }
            None => {
                let wallet = {
                    let mut last = self.last_assigned.lock();
                    let Some(wallet) = self.registry.pick_next(&last) else {
                        tracing::debug!(request_id = %request_id, "no miners connected right now, will retry dispatching this shard next tick");
                        return;
                    };
                    *last = Some(wallet.clone());
                    wallet
                };
                if snap.needs_walk {
                    if let Some(mut pipeline) = self.pipelines.get_mut(request_id) {
                        pipeline.moe_layer_walk = Some((snap.layer_start, wallet.clone()));
                    }
                    (wallet, Some(snap.layer_start))
                } else {
                    (wallet, None)
                }
            }
        };
        let Some(tx) = self.registry.get(&wallet) else { return };

        let assign = OpoiShardAssign {
            request_id: request_id.to_string(),
            model_id: snap.model_id,
            shard_index: snap.shard_index,
            layer_start: snap.layer_start,
            layer_end: snap.layer_end,
            total_layers: snap.total_layers,
            token_index: snap.token_index,
            max_tokens: snap.max_tokens,
            input: snap.current_input,
            model_gguf_url: source.gguf_url.clone(),
            model_gguf_sha256: snap.gguf_sha256,
            model_tokenizer_url: snap.is_entry.then(|| source.tokenizer_url.clone()),
            model_tokenizer_sha256: snap.is_entry.then(|| source.tokenizer_sha256.clone()),
            moe_layer_step,
        };
        self.assignments.insert(
            request_id.to_string(),
            ShardAssignment { wallet: wallet.clone(), shard_index: snap.shard_index, token_index: snap.token_index, moe_layer_step },
        );
        let _ = tx.send(build_shard_assign_line(&assign));
        tracing::info!(
            request_id = %request_id, shard_index = snap.shard_index, token_index = snap.token_index, wallet = %wallet, ?moe_layer_step,
            "dispatched shard"
        );
    }

    /// Called when a miner sends `opoi.shard_result`. Advances the pipeline
    /// (dispatches the next shard with the received tensor) or, on the last
    /// shard, submits on-chain and either starts the next token or finalizes.
    /// Named `*_inner` (not `handle_shard_result`) purely to avoid a
    /// same-name inherent-vs-trait-method ambiguity with the `ShardHandler`
    /// impl below, which is the actual public entry point.
    async fn handle_shard_result_inner(
        &self,
        wallet: &str,
        result: crate::opoi::wire::OpoiShardResult,
    ) -> Result<(), crate::error::AppError> {
        let request_id = result.request_id.clone();

        // F15-L: a request dispatched via SpeculativeEngine never has an
        // entry in THIS engine's own `pipelines`/`assignments` maps at
        // all — its `opoi.shard_result` traffic belongs entirely to that
        // engine's own state machine instead.
        if let Some(spec) = &self.speculative {
            if spec.has_pipeline(&request_id) {
                return spec.handle_shard_result(wallet, result).await;
            }
        }

        {
            let Some(assignment) = self.assignments.get(&request_id) else {
                return Err(crate::error::AppError::UnknownRequest);
            };
            if assignment.wallet != wallet
                || assignment.shard_index != result.shard_index
                || assignment.token_index != result.token_index
                || assignment.moe_layer_step != result.moe_layer_step
            {
                return Err(crate::error::AppError::NotAssignedToCaller);
            }
        }
        self.assignments.remove(&request_id);

        let model_id = { self.pipelines.get(&request_id).map(|p| p.manifest.model_id.clone()) };
        let Some(model_id) = model_id else { return Err(crate::error::AppError::UnknownRequest) };
        let Some(source) = self.model_sources.get(&model_id).cloned() else { return Err(crate::error::AppError::UnknownRequest) };

        // Payout-attribution bookkeeping (see `PipelineState::contributions`
        // doc comment): every accepted shard result is one real unit of
        // compute delivered by `wallet`, whether or not it turns out to be
        // the pipeline's last step (or, for a multi-layer walk, the
        // boundary step).
        if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
            *pipeline.contributions.entry(wallet.to_string()).or_insert(0) += 1;
        }

        // D2 multi-layer STATEFUL resume (2026-07-26 follow-up session):
        // if this result is ONE STEP of a per-layer walk (`moe_layer_step`
        // set) AND it's an INTERIOR layer (not this range's last layer),
        // this is NOT a submission point and does NOT advance `pos` — it
        // must be handled and returned BEFORE the on-chain submission
        // logic below, which only ever makes sense once per real dense
        // shard_index, not once per internal layer round trip. Falls
        // through to the EXISTING logic (unchanged) for every other case:
        // a plain dense shard (`moe_layer_step == None`), the OLD
        // whole-range-in-one-shot MoE path (also `None`), AND the
        // BOUNDARY layer of a new-protocol walk (`moe_layer_step ==
        // Some(layer_end - 1)`) — that last case is deliberately treated
        // exactly like the OLD path always was: `handle_expert_group_result`
        // below now takes an explicit `layer_idx` so it dispatches the
        // correct absolute layer either way.
        if let Some(layer_idx) = result.moe_layer_step {
            let layer_end = { self.pipelines.get(&request_id).map(|p| p.shards[p.pos].layer_end.unwrap_or(0)) };
            let Some(layer_end) = layer_end else { return Err(crate::error::AppError::UnknownRequest) };
            if layer_end == 0 || layer_idx + 1 < layer_end {
                let expert_ids = { self.pipelines.get(&request_id).and_then(|p| p.expert_group_for(p.pos).cloned()) };
                let Some(expert_ids) = expert_ids else {
                    tracing::error!(
                        request_id = %request_id, layer_idx,
                        "D2 multi-layer walk: result carried moe_layer_step but this shard's range has no registered EXPERT group — dropping pipeline"
                    );
                    self.pipelines.remove(&request_id);
                    return Ok(());
                };
                return self.handle_moe_interior_layer_result(&request_id, &source, expert_ids, layer_idx, &result).await;
            }
        }

        // Every shard gets exactly ONE on-chain submission per request — at
        // the end of the WHOLE generation (last token_index), never per
        // token step. The daemon only ever accepts the FIRST
        // submitshardresult for a given (request, shard, miner) and
        // silently ignores later ones (AddShardResult), so submitting
        // per-token would durably commit whichever intermediate hash
        // happened to land first — wrong for anything but max_tokens==1.
        // The miner side only reports a truly-finalized accumulated hash
        // when it's the last token (see shard_pool_client.rs's module doc)
        // — this check just decides WHEN to submit that hash on-chain.
        let is_last_token = {
            let pipeline = self.pipelines.get(&request_id);
            pipeline.map(|p| p.token_index + 1 >= p.max_tokens).unwrap_or(false)
        };

        if is_last_token {
            let submit = self.stake_pool.try_each(|addr| {
                let csd = self.csd.clone();
                let request_id = request_id.clone();
                let hash = result.output_hash.clone();
                async move { csd.submit_shard_result(&request_id, result.shard_index, &addr, &hash, 0).await }
            }).await;
            if let Err(e) = submit {
                tracing::warn!(request_id = %request_id, shard_index = result.shard_index, error = %e, "no pool address eligible to submit this shard result — pipeline stalls here until a retry mechanism exists (Sessão 3 scope)");
                return Ok(());
            }
        }

        let is_last_shard = {
            let pipeline = self.pipelines.get(&request_id);
            pipeline.map(|p| p.pos + 1 == p.shards.len()).unwrap_or(false)
        };

        // D2: does the shard that JUST completed have a registered EXPERT
        // group at its layer range? (see `PipelineState::expert_group_for`).
        // `None` for every pure-DENSE model (no EXPERT nodes in the graph at
        // all) — that's what keeps the plain dense-to-dense handoff below
        // byte-for-byte unchanged whenever this is `None`.
        let expert_ids = { self.pipelines.get(&request_id).and_then(|p| p.expert_group_for(p.pos).cloned()) };

        if !is_last_shard {
            if let Some(expert_ids) = expert_ids {
                // MoE fan-out step: this shard's output is the router
                // boundary hidden state for `expert_ids`, not directly the
                // next dense shard's input — see `handle_expert_group_result`.
                // `layer_idx`: `result.moe_layer_step` for a new-protocol
                // boundary result, or the implicit `layer_end - 1` the OLD
                // whole-range-in-one-shot path always operated on
                // (`.last()` of its nested `router_selection`) when
                // `moe_layer_step` is `None` — identical value either way.
                let layer_end = { self.pipelines.get(&request_id).map(|p| p.shards[p.pos].layer_end.unwrap_or(0)) }.unwrap_or(0);
                let layer_idx = result.moe_layer_step.unwrap_or(layer_end.saturating_sub(1));
                return self.handle_expert_group_result(&request_id, &source, expert_ids, layer_idx, &result).await;
            }
            // Advance to the next shard in the SAME token step, feeding it
            // this shard's real output tensor.
            if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
                pipeline.pos += 1;
                pipeline.current_input = ShardInputWire::Tensor { shape: result.output.shape.clone(), data_hex: result.output.data_hex.clone() };
                // See `PipelineState::moe_layer_walk`'s doc — always `None`
                // already on this plain dense-to-dense path, reset anyway
                // for defense in depth.
                pipeline.moe_layer_walk = None;
            }
            self.dispatch_current(&request_id, &source).await;
            return Ok(());
        }

        // D2 (narrower gap, see this file's module doc): the LAST dense
        // shard can have a registered EXPERT group too (every layer range
        // gets one when numExperts > 0), but there is no wire mechanism yet
        // to route a combined MoE output into a final norm+lm_head/
        // next-token decode step — no shard type exists for "run final
        // norm+lm_head over this externally-combined hidden state". Log it
        // clearly and fall back to the primary's own reported
        // `next_token_id` below, unchanged, same as a non-expert-routed
        // model — do NOT dispatch experts here (there is no consumer for
        // the combined output yet, so it would be wasted network work).
        if let Some(expert_ids) = &expert_ids {
            tracing::warn!(
                request_id = %request_id, shard_index = result.shard_index, registered_experts = expert_ids.len(),
                "D2: last dense shard's layer range has registered EXPERT nodes, but no wire mechanism exists yet to route a combined MoE output into final next-token decoding — falling back to the primary's own reported next_token_id unchanged"
            );
        }

        // Last shard of this token step. next_token_id must be present —
        // it's the whole point of reaching layer_end == total_layers.
        let Some(next_token_id) = result.next_token_id else {
            tracing::error!(request_id = %request_id, "last shard result carried no next_token_id — dropping pipeline");
            self.pipelines.remove(&request_id);
            return Ok(());
        };

        // This token step's real generated output, whether or not it's the
        // pipeline's last — accumulated here is what `finalize_pipeline`
        // eventually turns into the on-chain RESPONSE content.
        if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
            pipeline.generated_token_ids.push(next_token_id);
        }

        if is_last_token {
            tracing::info!(request_id = %request_id, "shard pipeline finished (max_tokens reached); driving commit/reveal/publish/payout");
            self.finalize_pipeline(&request_id).await;
            return Ok(());
        }

        // Start the next token: back to shard 0 (pos=0), token_index+1, fed
        // the newly-generated token id instead of a prompt.
        if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
            pipeline.pos = 0;
            pipeline.token_index += 1;
            pipeline.current_input = ShardInputWire::NextTokenId { token_id: next_token_id };
            pipeline.moe_layer_walk = None;
        }
        self.dispatch_current(&request_id, &source).await;
        Ok(())
    }

    /// D2: handles a dense shard's result when its layer range has a
    /// registered EXPERT group — the MoE counterpart of the plain
    /// dense-to-dense handoff `handle_shard_result_inner` otherwise does
    /// directly. `expert_ids` is the full set of experts registered for
    /// this range in the Model Execution Graph (NOT the active subset for
    /// any one layer — every layer in the range shares the same registered
    /// candidate set, see cloudservice's `BuildModelExecutionGraph`).
    ///
    /// Fans the ACTIVE subset out through `ExpertDispatcher::
    /// dispatch_and_join` and feeds the router-weighted-combined output
    /// forward as the next dense shard's input — exactly the same
    /// `pos += 1` / `current_input = Tensor {..}` / `dispatch_current` shape
    /// `handle_shard_result_inner`'s own dense-to-dense handoff already
    /// uses, just with the combined expert output standing in for a plain
    /// shard's `result.output`.
    ///
    /// D2 multi-layer STATEFUL resume (2026-07-26, later follow-up
    /// session): this method is now ONLY ever called for the RANGE'S LAST
    /// (boundary) LAYER's result — the caller (`handle_shard_result_inner`)
    /// intercepts and redirects any INTERIOR layer step to
    /// `handle_moe_interior_layer_result` before this method is ever
    /// reached, and never advances `pos`/submits on-chain for one (see that
    /// method's doc). This method's own logic is consequently unchanged in
    /// SUBSTANCE from before that split existed — it still validates and
    /// bounds-checks `router_selection`'s entries, still dispatches via
    /// `ExpertDispatcher::dispatch_and_join`, still advances `pos` and
    /// feeds the combined output forward as the next dense shard's input —
    /// the only two real changes are (1) it now takes an explicit
    /// `layer_idx: u32` (the ABSOLUTE layer being dispatched — either
    /// `result.moe_layer_step` for a new-protocol boundary result, or the
    /// implicit `layer_end - 1` the OLD whole-range-in-one-shot path always
    /// operated on, identical value either way) threaded through to
    /// `dispatch_and_join`'s new `layer_idx` parameter, and (2) it clears
    /// `pipeline.moe_layer_walk` before advancing `pos` (so the NEXT
    /// shard's dispatch decision is made fresh, not inherited).
    ///
    /// `router_selection` is `Vec<Vec<RouterExpertChoice>>` — for the OLD
    /// whole-range-in-one-shot path this has one entry per LAYER within the
    /// range (`.last()` is the boundary's); for the NEW per-layer-walk
    /// protocol it always has EXACTLY ONE entry (this layer's own
    /// selection — see `OpoiShardResult::moe_layer_step`'s doc comment for
    /// why the shape was deliberately kept the same so `.last()` needs no
    /// branch either way).
    async fn handle_expert_group_result(
        &self,
        request_id: &str,
        source: &ModelSource,
        expert_ids: Vec<u32>,
        layer_idx: u32,
        result: &crate::opoi::wire::OpoiShardResult,
    ) -> Result<(), crate::error::AppError> {
        // `router_selection` is the seam that carries the PRIMARY miner's
        // real per-layer router choice back to this bridge (see
        // `OpoiShardResult::router_selection`'s doc comment and this file's
        // module doc). Without it there is no sound way to know which of
        // `expert_ids` are actually active for this token, so — same
        // "stalls here until a retry mechanism exists" acceptance this file
        // already documents for submit failures — this pipeline simply
        // stalls at this step rather than guessing (e.g. dispatching ALL
        // registered experts with equal weight would silently compute a
        // mathematically different result than the real forward pass would
        // have produced, which is worse than stalling).
        let Some(per_layer_selections) = result.router_selection.as_ref() else {
            tracing::warn!(
                request_id = %request_id, shard_index = result.shard_index, registered_experts = expert_ids.len(),
                "this shard's layer range has registered EXPERT nodes but the shard result carried no router_selection — \
                 MoE distributed dispatch cannot proceed without it; pipeline stalls here until a retry/reassignment happens."
            );
            return Ok(());
        };

        if per_layer_selections.is_empty() || per_layer_selections.iter().any(|layer_sel| layer_sel.is_empty()) {
            tracing::error!(request_id = %request_id, "router_selection was present but empty (or had an empty layer entry) — dropping pipeline");
            self.pipelines.remove(request_id);
            return Ok(());
        }
        // Defensive bounds check, EVERY layer's entries — belt-and-
        // suspenders mirror of the daemon's own consensus-side
        // `IsExpertShardWithinManifestBounds` (cloudservice's
        // opoi_shard.h): never dispatch real network work for an
        // expert_id this range didn't actually register, at ANY layer.
        for layer_sel in per_layer_selections {
            for sel in layer_sel {
                if !expert_ids.contains(&sel.expert_id) {
                    tracing::error!(
                        request_id = %request_id, expert_id = sel.expert_id,
                        "router_selection names an expert_id not registered for this layer range in the Model Execution Graph — dropping pipeline"
                    );
                    self.pipelines.remove(request_id);
                    return Ok(());
                }
            }
        }

        if per_layer_selections.len() > 1 {
            // Only ever reachable via the OLD whole-range-in-one-shot path
            // now (the NEW per-layer-walk protocol always reports exactly
            // one entry per result — see this method's doc comment):
            // interior layers' selections are genuine, real, bounds-checked
            // router data, informational only for that OLD path (a future
            // Auditor sub-check (i) replay could use them), not turned into
            // external dispatch by IT specifically — the caller
            // (`handle_shard_result_inner`) routes every NEW-protocol
            // interior-layer result to `handle_moe_interior_layer_result`
            // instead, which DOES dispatch them.
            tracing::info!(
                request_id = %request_id, shard_index = result.shard_index, interior_layers = per_layer_selections.len() - 1,
                "D2 (OLD whole-range-in-one-shot path): {} interior layer(s)' real router selections received and validated, but not \
                 externally dispatched by this path (still local-combine only on the primary) — only the range's last layer is dispatched below",
                per_layer_selections.len() - 1
            );
        }

        // The range's LAST (boundary) layer's selection — the ONLY entry
        // for the NEW per-layer-walk protocol, the LAST of several for the
        // OLD whole-range-in-one-shot path. See this method's doc comment.
        let selections = per_layer_selections.last().expect("checked non-empty above");

        let Ok(input_values) = hex_to_f32(&result.output.data_hex) else {
            tracing::error!(request_id = %request_id, "shard result output data_hex failed to decode as an f32 tensor — dropping pipeline");
            self.pipelines.remove(request_id);
            return Ok(());
        };

        let Some((layer_start, layer_end, token_index, model_id, gguf_sha256)) = self.pipelines.get(request_id).map(|p| {
            let shard = &p.shards[p.pos];
            (shard.layer_start.unwrap_or(0), shard.layer_end.unwrap_or(0), p.token_index, p.manifest.model_id.clone(), p.manifest.backbone_pom_root.clone())
        }) else {
            return Err(crate::error::AppError::UnknownRequest);
        };

        let experts: Vec<SelectedExpert> = selections
            .iter()
            .map(|sel| SelectedExpert { expert_id: sel.expert_id, weight: sel.weight, shape: result.output.shape.clone(), data: input_values.clone() })
            .collect();

        // D2 follow-up: real per-expert VRAM floor, read straight from the
        // GGUF's own header (see `expert_vram.rs`'s module doc for the full
        // mechanism/rationale) — every expert of a given layer shares the
        // same FFN-expert tensor shape/dtype (see that module's doc comment
        // for why `min_vram_mb_for` below can safely ignore its `expert_id`
        // argument), so this is resolved to a single `u64` up front rather
        // than varying per expert. `unwrap_or(0)` on failure (network error,
        // header not parseable, unsupported dtype, ...) falls back to the
        // pre-follow-up behavior — no floor at all — rather than stalling
        // the whole MoE dispatch over an unavailable VRAM estimate.
        // Per-LAYER VRAM estimate now (`layer_idx, layer_idx + 1`), not the
        // whole owning range — more precise once every layer is dispatched
        // individually (each layer's experts can have a different real
        // size), and correct either way since `min_vram_mb_for_range`
        // already takes an arbitrary sub-range.
        let min_vram_mb = self
            .expert_vram
            .min_vram_mb_for_range(&model_id, &source.gguf_url, &gguf_sha256, layer_idx, layer_idx + 1, expert_ids.len() as u64)
            .await
            .unwrap_or(0);

        let combined = self
            .expert_dispatcher
            .dispatch_and_join(request_id, layer_start, layer_end, layer_idx, token_index, &model_id, &source.gguf_url, &gguf_sha256, experts, move |_expert_id| min_vram_mb)
            .await;

        match combined {
            Ok(output) => {
                if let Some(mut pipeline) = self.pipelines.get_mut(request_id) {
                    pipeline.pos += 1;
                    pipeline.current_input = ShardInputWire::Tensor { shape: output.shape, data_hex: output.data_hex };
                    // See `PipelineState::moe_layer_walk`'s doc — this
                    // shard's walk (if any) just finished; the next
                    // shard's dispatch decision must be made fresh.
                    pipeline.moe_layer_walk = None;
                }
                self.dispatch_current(request_id, source).await;
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %request_id, error = %e,
                    "D2 expert fan-out/join failed for this layer range — pipeline stalls here until a retry mechanism exists (same acceptance as this file's other stall cases)"
                );
                Ok(())
            }
        }
    }

    /// D2 multi-layer STATEFUL resume (2026-07-26 follow-up session): the
    /// NEW-protocol counterpart of `handle_expert_group_result` above, for
    /// an INTERIOR layer of a multi-layer range walk (`layer_idx + 1 <
    /// layer_end` — NOT this range's last layer). Structurally almost
    /// identical to that method (same validation, same
    /// `ExpertDispatcher::dispatch_and_join` call, same combine), with two
    /// deliberate differences that follow directly from the on-chain
    /// submission-timing design decision this session made (see this
    /// crate's D2 session report, "Choice A" — multiple internal wire
    /// round trips per real dense shard_index, exactly ONE on-chain
    /// submission at the end):
    ///   1. Does NOT touch on-chain submission at all — an interior
    ///      layer's result is never what gets submitted (only the
    ///      boundary layer's `result.output_hash`/`result.output` is,
    ///      exactly as before this session).
    ///   2. Does NOT advance `pos` — this dense shard/range isn't done
    ///      yet. Instead it feeds the combined output back to the SAME
    ///      PINNED wallet (`pipeline.moe_layer_walk`) as the next resume
    ///      step's input, advancing `moe_layer_walk`'s `next_layer_idx`
    ///      by one and re-invoking `dispatch_current` (which will build
    ///      the resume `opoi.shard_assign` — see that method's doc).
    ///
    /// `router_selection` is expected to have EXACTLY ONE entry (this
    /// layer's own selection) — see `OpoiShardResult::moe_layer_step`'s doc
    /// comment. Uses `.last()` (not `.first()`/direct indexing) purely to
    /// keep the exact same "any layer entry, however many there happen to
    /// be" defensive shape `handle_expert_group_result` already uses,
    /// though in practice there is only ever the one.
    async fn handle_moe_interior_layer_result(
        &self,
        request_id: &str,
        source: &ModelSource,
        expert_ids: Vec<u32>,
        layer_idx: u32,
        result: &crate::opoi::wire::OpoiShardResult,
    ) -> Result<(), crate::error::AppError> {
        let Some(per_layer_selections) = result.router_selection.as_ref() else {
            tracing::warn!(
                request_id = %request_id, shard_index = result.shard_index, layer_idx, registered_experts = expert_ids.len(),
                "D2 multi-layer walk: this layer's result carried no router_selection — pipeline stalls here until a retry/reassignment happens."
            );
            return Ok(());
        };
        if per_layer_selections.is_empty() || per_layer_selections.iter().any(|layer_sel| layer_sel.is_empty()) {
            tracing::error!(request_id = %request_id, layer_idx, "router_selection was present but empty (or had an empty layer entry) — dropping pipeline");
            self.pipelines.remove(request_id);
            return Ok(());
        }
        for layer_sel in per_layer_selections {
            for sel in layer_sel {
                if !expert_ids.contains(&sel.expert_id) {
                    tracing::error!(
                        request_id = %request_id, expert_id = sel.expert_id, layer_idx,
                        "router_selection names an expert_id not registered for this layer range in the Model Execution Graph — dropping pipeline"
                    );
                    self.pipelines.remove(request_id);
                    return Ok(());
                }
            }
        }
        let selections = per_layer_selections.last().expect("checked non-empty above");

        let Ok(input_values) = hex_to_f32(&result.output.data_hex) else {
            tracing::error!(request_id = %request_id, layer_idx, "shard result output data_hex failed to decode as an f32 tensor — dropping pipeline");
            self.pipelines.remove(request_id);
            return Ok(());
        };

        let Some((layer_start, layer_end, token_index, model_id, gguf_sha256)) = self.pipelines.get(request_id).map(|p| {
            let shard = &p.shards[p.pos];
            (shard.layer_start.unwrap_or(0), shard.layer_end.unwrap_or(0), p.token_index, p.manifest.model_id.clone(), p.manifest.backbone_pom_root.clone())
        }) else {
            return Err(crate::error::AppError::UnknownRequest);
        };

        let experts: Vec<SelectedExpert> = selections
            .iter()
            .map(|sel| SelectedExpert { expert_id: sel.expert_id, weight: sel.weight, shape: result.output.shape.clone(), data: input_values.clone() })
            .collect();

        let min_vram_mb = self
            .expert_vram
            .min_vram_mb_for_range(&model_id, &source.gguf_url, &gguf_sha256, layer_idx, layer_idx + 1, expert_ids.len() as u64)
            .await
            .unwrap_or(0);

        let combined = self
            .expert_dispatcher
            .dispatch_and_join(request_id, layer_start, layer_end, layer_idx, token_index, &model_id, &source.gguf_url, &gguf_sha256, experts, move |_expert_id| min_vram_mb)
            .await;

        match combined {
            Ok(output) => {
                if let Some(mut pipeline) = self.pipelines.get_mut(request_id) {
                    pipeline.current_input = ShardInputWire::Tensor { shape: output.shape, data_hex: output.data_hex };
                    // Advance the walk to the NEXT layer, staying pinned to
                    // the SAME wallet — see `PipelineState::moe_layer_walk`'s
                    // doc for why this must never round-robin. `pos` is
                    // deliberately NOT touched — this dense shard isn't done.
                    if let Some((_, pinned_wallet)) = pipeline.moe_layer_walk.clone() {
                        pipeline.moe_layer_walk = Some((layer_idx + 1, pinned_wallet));
                    }
                }
                self.dispatch_current(request_id, source).await;
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %request_id, layer_idx, error = %e,
                    "D2 multi-layer walk: interior layer's expert fan-out/join failed — pipeline stalls here until a retry mechanism exists (same acceptance as this file's other stall cases)"
                );
                Ok(())
            }
        }
    }

    /// Drives a just-completed shard pipeline's response through the SAME
    /// commit -> reveal -> publish -> payout lifecycle `OpoiEngine` already
    /// has working for whole-model responses, by writing the same
    /// `opoi_submissions` row shape it does and handing off to its shared
    /// `do_commit` (see that function's doc comment) — reveal/publish
    /// (`OpoiEngine::poll_reveal_tick`/`do_publish`) and payout
    /// (`payout::payout_tick`) both already operate purely off DB rows, not
    /// off which engine created them, so nothing else needs to change for
    /// those to pick this row up on their normal interval.
    ///
    /// Two deliberate simplifications, both documented here rather than
    /// solved elaborately (see this task's scope note — matching existing
    /// semantics safely beats inventing new ones):
    ///
    /// 1. **`response_hex` has no human-readable text behind it.** Unlike
    ///    `OpoiEngine`, whose miners decode and hand back real text
    ///    directly, this bridge has no tokenizer of its own (only the
    ///    per-shard miners do, via their own GGUF vocab) — there is no
    ///    on-bridge way to detokenize `generated_token_ids` into text. The
    ///    on-chain commit/reveal/publish path only actually requires
    ///    `response_hash == sha256(response_hex)` self-consistency (see
    ///    `OpoiEngine::handle_submit_result`'s step 3 and this file's
    ///    `sha256_hex` helper) — it never interprets the bytes — so
    ///    `response_hex` here is instead the accumulated generated token
    ///    ids, each encoded as 4 bytes little-endian, in generation order.
    ///    This satisfies that requirement and is fully reproducible, but a
    ///    downstream consumer that expects `submitopoicontent`'s published
    ///    RESPONSE bytes to be human-readable text for a shard-routed
    ///    request will need a detokenize step of its own (future work).
    ///
    /// 2. **Payout attribution picks ONE wallet, not a fair split.** A dense
    ///    pipeline can genuinely involve several different downstream
    ///    wallets (`dispatch_current` round-robins every shard dispatch
    ///    across whoever is connected at that instant), but
    ///    `opoi_submissions` only has room for one `miner_wallet` per row,
    ///    and that's a hard constraint (one row = one on-chain
    ///    commit/reveal/publish per `request_id`, enforced by
    ///    `uq_opoi_submissions_active_request` — not just a DB nicety), not
    ///    something this change should relax. `OpoiEngine`'s own existing
    ///    semantics are actually already "single assignee gets 100%" (there
    ///    is no N-way payment split anywhere in the whole-model path
    ///    either — the stake-pool addresses used for commit/reveal are the
    ///    bridge's own custodied identities, unrelated to downstream miner
    ///    payment). Extending that same single-attribution model here as
    ///    conservatively as possible: the wallet that delivered the MOST
    ///    accepted shard-result steps for this request (see
    ///    `PipelineState::contributions`) gets the full reward; ties break
    ///    on wallet address ordering (arbitrary but deterministic). A fair
    ///    proportional split across every contributing wallet would need
    ///    either a schema change (a per-request payout-split table) or
    ///    multiple payout rows keyed some other way than `request_id` —
    ///    real future work, not a "guess elaborately" call for this pass.
    async fn finalize_pipeline(&self, request_id: &str) {
        let Some((_, pipeline)) = self.pipelines.remove(request_id) else { return };

        let Some(winning_wallet) = select_winning_wallet(&pipeline.contributions) else {
            tracing::error!(request_id = %request_id, "shard pipeline finished with no recorded contributions; cannot attribute payout, dropping");
            return;
        };

        let (response_hash, response_hex) = build_response(&pipeline.generated_token_ids);
        let token_count = pipeline.generated_token_ids.len() as u32;

        if let Some(b3lite_cfg) = &self.b3lite {
            self.record_b3lite_receipt(b3lite_cfg, request_id, &winning_wallet, &pipeline, &response_hash).await;
        }

        match db::repo::create_submission(
            &self.db,
            &winning_wallet,
            request_id,
            Some(&pipeline.manifest.model_id),
            Some(&pipeline.prompt_hash),
            None,
            None,
            &response_hash,
            &response_hex,
            token_count as i32,
        )
        .await
        {
            Ok(id) => {
                tracing::info!(
                    submission_id = id, request_id = %request_id, wallet = %winning_wallet, token_count,
                    "shard pipeline response recorded; driving commit/reveal/publish (shared with OpoiEngine)"
                );
                let csd = self.csd.clone();
                let db_pool = self.db.clone();
                let stake_pool = self.stake_pool.clone();
                let request_id = request_id.to_string();
                tokio::spawn(async move {
                    do_commit(csd, db_pool, stake_pool, id, request_id, response_hash, token_count).await;
                });
            }
            Err(e) => {
                tracing::error!(error = %e, request_id = %request_id, "failed to persist opoi_submissions row for completed shard pipeline; payout lost for this request");
            }
        }
    }

    /// B3-lite (see `b3lite.rs`'s module doc): signs and persists a served-
    /// response receipt for a just-finished manifest-pinned pipeline, and
    /// decides whether it gets queued for a real Auditor replay. Best-
    /// effort — a failure here only means this response wasn't recorded/
    /// sampled for B3-lite, never blocks the on-chain commit/reveal/publish
    /// path `finalize_pipeline` drives regardless (see the B3-lite scope
    /// doc: no consensus change, this is purely additional off-chain
    /// bookkeeping).
    async fn record_b3lite_receipt(
        &self,
        cfg: &B3LiteConfig,
        request_id: &str,
        miner_wallet: &str,
        pipeline: &PipelineState,
        response_hash: &str,
    ) {
        let gguf_sha256 = &pipeline.manifest.backbone_pom_root;
        let generated_token_ids_hex = {
            let mut raw = Vec::with_capacity(pipeline.generated_token_ids.len() * 4);
            for id in &pipeline.generated_token_ids {
                raw.extend_from_slice(&id.to_le_bytes());
            }
            hex::encode(raw)
        };

        let fields = b3lite::ReceiptFields {
            request_id,
            miner_wallet,
            model_id: &pipeline.manifest.model_id,
            gguf_sha256,
            response_hash,
            generated_token_ids: &pipeline.generated_token_ids,
        };
        let signature = b3lite::sign_receipt(&cfg.secret, &fields);
        let sampled = b3lite::should_sample(&signature, cfg.sample_rate);

        match db::repo::create_b3lite_receipt(
            &self.db,
            request_id,
            miner_wallet,
            &pipeline.manifest.model_id,
            gguf_sha256,
            Some(&pipeline.prompt_hash),
            &pipeline.original_prompt_hex,
            response_hash,
            &generated_token_ids_hex,
            pipeline.manifest.num_layers as i32,
            &signature,
            sampled,
        )
        .await
        {
            Ok(id) => {
                tracing::info!(receipt_id = id, request_id = %request_id, sampled, "B3-lite receipt recorded");
            }
            Err(e) => {
                tracing::warn!(error = %e, request_id = %request_id, "failed to persist B3-lite receipt (non-fatal — commit/reveal/publish proceeds regardless)");
            }
        }
    }
}

#[async_trait::async_trait]
impl crate::opoi::handler::ShardHandler for ShardEngine {
    async fn handle_shard_result(&self, wallet: &str, result: crate::opoi::wire::OpoiShardResult) -> Result<(), crate::error::AppError> {
        self.handle_shard_result_inner(wallet, result).await
    }

    async fn on_disconnect(&self, wallet: &str) {
        self.assignments.retain(|_, a| a.wallet != wallet);
        // NOTE: `SpeculativeEngine::on_disconnect` is NOT forwarded from
        // here — `proxy/session.rs` already calls it directly (it holds its
        // own `Arc<dyn SpeculativeHandler>`, wired up alongside this
        // engine's own `Arc<dyn ShardHandler>` in main.rs), so forwarding
        // it here too would just be a harmless-but-redundant double call.
    }
}

/// Used by `finalize_pipeline` to hash the accumulated generated-token-id
/// bytes into the on-chain `response_hash` (hashing isn't otherwise this
/// module's job — `shard_compute.rs` on the miner side computes each
/// shard's own output hash).
fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// D2: pure MEG-splitting logic extracted out of `poll_and_start_tick` so
/// it's testable without a live `csd` connection. Separates a fetched Model
/// Execution Graph into its DENSE nodes — sorted by `shard_index`, walked
/// exactly as before by `dispatch_current`/`handle_shard_result_inner` — and
/// its `EXPERT` nodes, grouped by the `(layer_start, layer_end)` range of
/// the DENSE shard they sit alongside (see cloudservice's
/// `BuildModelExecutionGraph`: every DENSE node for every layer range first,
/// in range order, then every EXPERT node grouped by range in that same
/// order — `numExperts` nodes per range). A `ShardDescriptor` with no
/// `layer_start`/`layer_end` (shouldn't happen for a real EXPERT node, but
/// defensively handled) defaults both to `0`. A pure-DENSE model (no
/// `EXPERT` entries in the graph at all) yields an empty `expert_groups`
/// map — exactly what keeps that pipeline byte-for-byte unchanged
/// (`PipelineState::expert_group_for` always returns `None`).
fn split_model_execution_graph(shards: &[ShardDescriptor]) -> (Vec<ShardDescriptor>, HashMap<(u32, u32), Vec<u32>>) {
    let mut dense: Vec<ShardDescriptor> = shards.iter().filter(|s| s.shard_type == "DENSE").cloned().collect();
    dense.sort_by_key(|s| s.shard_index);

    let mut expert_groups: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for s in shards.iter().filter(|s| s.shard_type == "EXPERT") {
        let key = (s.layer_start.unwrap_or(0), s.layer_end.unwrap_or(0));
        expert_groups.entry(key).or_default().push(s.expert_id.unwrap_or(0));
    }

    (dense, expert_groups)
}

/// Pure payout-attribution decision, extracted out of `finalize_pipeline` so
/// it's testable without a live `PgPool`/`CsdRpcClient`. Picks whichever
/// wallet delivered the most accepted shard-result steps (see
/// `PipelineState::contributions`'s doc comment for why "most steps" is the
/// attribution rule). `BTreeMap::iter()` yields entries in ascending key
/// order, and `Iterator::max_by_key` keeps the LAST element it sees among
/// ties — so when two or more wallets are tied for the highest count, the
/// lexicographically GREATEST wallet address wins. That ordering is exactly
/// what "arbitrary but deterministic" (finalize_pipeline's doc comment,
/// simplification #2) refers to; this function and its tests pin the exact
/// direction down. Returns `None` only when `contributions` is empty (a
/// pipeline that finished with zero recorded contributions).
fn select_winning_wallet(contributions: &BTreeMap<String, u32>) -> Option<String> {
    contributions.iter().max_by_key(|(_, count)| **count).map(|(wallet, _)| wallet.clone())
}

/// Pure construction of the on-chain `(response_hash, response_hex)` pair
/// from a pipeline's accumulated generated token ids, extracted out of
/// `finalize_pipeline` so it's testable without a live `PgPool`. See
/// `finalize_pipeline`'s doc comment (simplification #1) for why the byte
/// encoding is each token id as 4 bytes little-endian, in generation order,
/// rather than human-readable text.
fn build_response(token_ids: &[u32]) -> (String /* response_hash */, String /* response_hex */) {
    let mut raw = Vec::with_capacity(token_ids.len() * 4);
    for token_id in token_ids {
        raw.extend_from_slice(&token_id.to_le_bytes());
    }
    (sha256_hex(&raw), hex::encode(&raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- select_winning_wallet -------------------------------------------------

    #[test]
    fn winning_wallet_none_when_no_contributions() {
        let contributions: BTreeMap<String, u32> = BTreeMap::new();
        assert_eq!(select_winning_wallet(&contributions), None);
    }

    #[test]
    fn winning_wallet_single_contributor_always_wins() {
        let mut contributions = BTreeMap::new();
        contributions.insert("wallet_only".to_string(), 1);
        assert_eq!(select_winning_wallet(&contributions), Some("wallet_only".to_string()));
    }

    #[test]
    fn winning_wallet_clear_majority_wins() {
        let mut contributions = BTreeMap::new();
        contributions.insert("wallet_a".to_string(), 3);
        contributions.insert("wallet_b".to_string(), 9);
        contributions.insert("wallet_c".to_string(), 1);
        assert_eq!(select_winning_wallet(&contributions), Some("wallet_b".to_string()));
    }

    /// Pins down the exact tie-break direction: among wallets tied for the
    /// highest step count, the lexicographically GREATEST address wins
    /// (falls out of BTreeMap's ascending iteration + max_by_key keeping the
    /// last-seen max). This is the behavior finalize_pipeline's doc comment
    /// calls "arbitrary but deterministic" — this test makes the direction
    /// explicit and regression-proof.
    #[test]
    fn winning_wallet_tie_breaks_toward_greatest_address() {
        let mut contributions = BTreeMap::new();
        contributions.insert("aaa_wallet".to_string(), 5);
        contributions.insert("zzz_wallet".to_string(), 5);
        contributions.insert("mmm_wallet".to_string(), 1); // not tied, should never win
        assert_eq!(select_winning_wallet(&contributions), Some("zzz_wallet".to_string()));
    }

    #[test]
    fn winning_wallet_three_way_tie_breaks_toward_greatest_address() {
        let mut contributions = BTreeMap::new();
        contributions.insert("aaa".to_string(), 5);
        contributions.insert("bbb".to_string(), 5);
        contributions.insert("ccc".to_string(), 5);
        assert_eq!(select_winning_wallet(&contributions), Some("ccc".to_string()));
    }

    /// Confirms the tie-break is genuinely a property of key ordering, not
    /// insertion order (BTreeMap always iterates by key regardless of
    /// insertion sequence, but this makes that assumption explicit and
    /// would catch a regression to an insertion-ordered map type).
    #[test]
    fn winning_wallet_tie_break_independent_of_insertion_order() {
        let mut contributions = BTreeMap::new();
        contributions.insert("zzz_wallet".to_string(), 5);
        contributions.insert("aaa_wallet".to_string(), 5);
        assert_eq!(select_winning_wallet(&contributions), Some("zzz_wallet".to_string()));
    }

    #[test]
    fn winning_wallet_deterministic_across_repeated_calls() {
        let mut contributions = BTreeMap::new();
        contributions.insert("wallet_x".to_string(), 7);
        contributions.insert("wallet_y".to_string(), 7);
        let first = select_winning_wallet(&contributions);
        for _ in 0..20 {
            assert_eq!(select_winning_wallet(&contributions), first);
        }
    }

    // ---- build_response ---------------------------------------------------------

    #[test]
    fn build_response_empty_tokens_is_empty_hex_of_empty_hash() {
        let (hash, hex_str) = build_response(&[]);
        assert_eq!(hex_str, "");
        // Cross-checked independently against Sha256 over an empty input,
        // not hardcoded, so this can't drift from the real algorithm.
        let expected = hex::encode(Sha256::digest([]));
        assert_eq!(hash, expected);
    }

    #[test]
    fn build_response_is_deterministic_for_same_tokens() {
        let tokens = vec![1u32, 2, 3, 42, 999_999];
        let (hash1, hex1) = build_response(&tokens);
        let (hash2, hex2) = build_response(&tokens);
        assert_eq!(hash1, hash2);
        assert_eq!(hex1, hex2);
    }

    #[test]
    fn build_response_is_sensitive_to_token_differences() {
        let (hash_a, hex_a) = build_response(&[1, 2, 3]);
        let (hash_b, hex_b) = build_response(&[1, 2, 4]);
        assert_ne!(hash_a, hash_b);
        assert_ne!(hex_a, hex_b);
    }

    /// Order matters — the token sequence is generation order, and a
    /// different order is a different response, so it must hash differently.
    #[test]
    fn build_response_is_sensitive_to_token_order() {
        let (hash_a, hex_a) = build_response(&[1, 2, 3]);
        let (hash_b, hex_b) = build_response(&[3, 2, 1]);
        assert_ne!(hash_a, hash_b);
        assert_ne!(hex_a, hex_b);
    }

    /// Pins the exact byte encoding down: each token id as 4 bytes
    /// little-endian, concatenated in generation order — per
    /// finalize_pipeline's doc comment. A change to big-endian, varint, or
    /// any other encoding would break this without necessarily breaking
    /// determinism/sensitivity tests above.
    #[test]
    fn build_response_hex_matches_expected_little_endian_encoding() {
        let tokens = vec![1u32, 256, 65536, u32::MAX];
        let (hash, hex_str) = build_response(&tokens);

        let mut expected_raw = Vec::new();
        for t in &tokens {
            expected_raw.extend_from_slice(&t.to_le_bytes());
        }
        let expected_hex = hex::encode(&expected_raw);
        assert_eq!(hex_str, expected_hex);
        assert_eq!(hex_str.len(), tokens.len() * 8); // 4 bytes -> 8 hex chars per token

        // response_hash must be sha256 of exactly the decoded response_hex
        // bytes — this is the self-consistency property the on-chain
        // commit/reveal path actually depends on (see finalize_pipeline's
        // doc comment): response_hash == sha256(response_hex-decoded-bytes).
        let decoded = hex::decode(&hex_str).unwrap();
        assert_eq!(decoded, expected_raw);
        let expected_hash = hex::encode(Sha256::digest(&expected_raw));
        assert_eq!(hash, expected_hash);
    }

    #[test]
    fn build_response_single_token_round_trips() {
        let (hash, hex_str) = build_response(&[0x1234_5678]);
        assert_eq!(hex_str, "78563412"); // little-endian byte order
        let expected_hash = hex::encode(Sha256::digest([0x78, 0x56, 0x34, 0x12]));
        assert_eq!(hash, expected_hash);
    }

    #[test]
    fn build_response_length_scales_with_token_count() {
        let tokens: Vec<u32> = (0..50).collect();
        let (_, hex_str) = build_response(&tokens);
        assert_eq!(hex_str.len(), tokens.len() * 8);
    }

    // ---- split_model_execution_graph (D2) ---------------------------------------

    fn dense(shard_index: u32, layer_start: u32, layer_end: u32) -> ShardDescriptor {
        ShardDescriptor { shard_index, shard_type: "DENSE".to_string(), layer_start: Some(layer_start), layer_end: Some(layer_end), expert_id: Some(0) }
    }

    fn expert(shard_index: u32, layer_start: u32, layer_end: u32, expert_id: u32) -> ShardDescriptor {
        ShardDescriptor {
            shard_index,
            shard_type: "EXPERT".to_string(),
            layer_start: Some(layer_start),
            layer_end: Some(layer_end),
            expert_id: Some(expert_id),
        }
    }

    #[test]
    fn split_pure_dense_graph_yields_empty_expert_groups() {
        let graph = vec![dense(0, 0, 4), dense(1, 4, 8)];
        let (shards, expert_groups) = split_model_execution_graph(&graph);
        assert_eq!(shards.len(), 2);
        assert!(expert_groups.is_empty());
    }

    #[test]
    fn split_dense_only_graph_sorts_by_shard_index_even_if_input_unsorted() {
        let graph = vec![dense(1, 4, 8), dense(0, 0, 4)];
        let (shards, _) = split_model_execution_graph(&graph);
        assert_eq!(shards[0].shard_index, 0);
        assert_eq!(shards[1].shard_index, 1);
    }

    /// Mirrors cloudservice's `BuildModelExecutionGraph` MoE layout exactly:
    /// every DENSE node for every layer range first (in range order), then
    /// every EXPERT node grouped by range in that same order, `numExperts`
    /// nodes per range.
    #[test]
    fn split_moe_graph_groups_experts_by_layer_range() {
        let graph = vec![
            dense(0, 0, 4),
            dense(1, 4, 8),
            expert(2, 0, 4, 0),
            expert(3, 0, 4, 1),
            expert(4, 4, 8, 0),
            expert(5, 4, 8, 1),
        ];
        let (shards, expert_groups) = split_model_execution_graph(&graph);

        assert_eq!(shards.len(), 2);
        assert!(shards.iter().all(|s| s.shard_type == "DENSE"));

        assert_eq!(expert_groups.len(), 2);
        let mut range0 = expert_groups.get(&(0, 4)).expect("range [0,4) has a group").clone();
        range0.sort();
        assert_eq!(range0, vec![0, 1]);
        let mut range1 = expert_groups.get(&(4, 8)).expect("range [4,8) has a group").clone();
        range1.sort();
        assert_eq!(range1, vec![0, 1]);
    }

    #[test]
    fn split_graph_with_no_dense_nodes_yields_empty_dense_list() {
        let graph = vec![expert(0, 0, 4, 0), expert(1, 0, 4, 1)];
        let (shards, expert_groups) = split_model_execution_graph(&graph);
        assert!(shards.is_empty());
        assert_eq!(expert_groups.len(), 1);
    }

    #[test]
    fn split_expert_node_with_missing_layer_bounds_defaults_key_to_zero_zero() {
        let malformed = ShardDescriptor { shard_index: 1, shard_type: "EXPERT".to_string(), layer_start: None, layer_end: None, expert_id: Some(3) };
        let (_, expert_groups) = split_model_execution_graph(&[malformed]);
        assert_eq!(expert_groups.get(&(0, 0)), Some(&vec![3]));
    }

    // ---- PipelineState::expert_group_for (D2) -----------------------------------

    fn test_manifest(model_id: &str, num_layers: u32) -> ModelManifest {
        ModelManifest { model_id: model_id.to_string(), arch_type: "MOE".to_string(), num_layers, backbone_pom_root: "gguf-hash".to_string(), status: "ACTIVE".to_string() }
    }

    fn test_pipeline(shards: Vec<ShardDescriptor>, expert_groups: HashMap<(u32, u32), Vec<u32>>) -> PipelineState {
        PipelineState {
            shards,
            manifest: test_manifest("model-x", 8),
            prompt_hash: "ph".to_string(),
            max_tokens: 4,
            pos: 0,
            token_index: 0,
            current_input: ShardInputWire::Prompt { prompt_hex: "aa".to_string() },
            original_prompt_hex: "aa".to_string(),
            generated_token_ids: Vec::new(),
            contributions: BTreeMap::new(),
            expert_groups,
            moe_layer_walk: None,
        }
    }

    #[test]
    fn expert_group_for_none_when_no_groups_registered_at_all() {
        let pipeline = test_pipeline(vec![dense(0, 0, 4), dense(1, 4, 8)], HashMap::new());
        assert!(pipeline.expert_group_for(0).is_none());
        assert!(pipeline.expert_group_for(1).is_none());
    }

    #[test]
    fn expert_group_for_some_when_this_shards_range_has_a_group() {
        let mut groups = HashMap::new();
        groups.insert((0, 4), vec![0, 1]);
        let pipeline = test_pipeline(vec![dense(0, 0, 4), dense(1, 4, 8)], groups);
        assert_eq!(pipeline.expert_group_for(0), Some(&vec![0, 1]));
        assert!(pipeline.expert_group_for(1).is_none());
    }

    #[test]
    fn expert_group_for_none_when_group_is_present_but_empty() {
        let mut groups = HashMap::new();
        groups.insert((0, 4), Vec::<u32>::new());
        let pipeline = test_pipeline(vec![dense(0, 0, 4)], groups);
        assert!(pipeline.expert_group_for(0).is_none());
    }

    #[test]
    fn expert_group_for_out_of_range_pos_is_none() {
        let pipeline = test_pipeline(vec![dense(0, 0, 4)], HashMap::new());
        assert!(pipeline.expert_group_for(5).is_none());
    }
}

#[cfg(test)]
mod d2_expert_group_dispatch_tests {
    //! End-to-end (in-process, no live network/DB) tests for the actual
    //! wiring this session's task adds: `ShardEngine::handle_shard_result_inner`
    //! recognizing an EXPERT-graph layer range and routing to
    //! `ExpertDispatcher::dispatch_and_join`, then feeding the combined
    //! output forward as the next dense shard's input — exactly the
    //! integration point described in `shard_engine.rs`'s module doc.
    //!
    //! `ShardEngine` is constructed with a lazily-connecting `PgPool`
    //! (`connect_lazy` never touches the network) and a `CsdRpcClient`
    //! pointed at a bogus URL — neither is ever actually called by the code
    //! path under test (`handle_expert_group_result`/`dispatch_current` only
    //! touch `self.pipelines`/`self.registry`/`self.expert_dispatcher`), so
    //! this needs no live Postgres/csd, same spirit as `expert_dispatch.rs`'s
    //! own `dispatch_tests` module (fake channels + a real `MinerRegistry`,
    //! no live network).
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::{ModelSource, ModelSourceConfig, PipelineState, ShardAssignment, ShardEngine};
    use crate::miner_registry::MinerRegistry;
    use crate::opoi::expert_dispatch::{hex_to_f32, ExpertDispatcher};
    use crate::opoi::handler::ExpertHandler;
    use crate::opoi::wire::{ExpertOutputWire, OpoiExpertResult, OpoiShardResult, RouterExpertChoice, ShardInputWire, ShardOutputWire};
    use crate::rpc::types::{ModelManifest, ShardDescriptor};
    use crate::rpc::CsdRpcClient;
    use crate::stake_pool::StakePool;

    fn f32_to_hex(data: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        hex::encode(bytes)
    }

    fn dense(shard_index: u32, layer_start: u32, layer_end: u32) -> ShardDescriptor {
        ShardDescriptor { shard_index, shard_type: "DENSE".to_string(), layer_start: Some(layer_start), layer_end: Some(layer_end), expert_id: Some(0) }
    }

    /// Neither `csd` nor `db` is ever actually reached by
    /// `handle_shard_result_inner`'s expert-group branch or by
    /// `dispatch_current` — see this module's doc comment.
    fn test_engine(registry: Arc<MinerRegistry>, model_id: &str, expert_dispatcher: Arc<ExpertDispatcher>) -> ShardEngine {
        let csd = Arc::new(CsdRpcClient::new("http://127.0.0.1:1".to_string(), "u".to_string(), "p".to_string()));
        let db = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgres://user:pass@127.0.0.1/db")
            .expect("connect_lazy never touches the network, only parses the URL");
        // `StakePool::new` asserts a non-empty address list — this bogus
        // address is never actually eligible for anything real (no live
        // `csd`), it exists purely so construction doesn't panic. Only
        // `last_shard_with_expert_group_falls_back_to_primarys_next_token_id`
        // below ever reaches a code path that calls into it at all.
        let stake_pool = Arc::new(StakePool::new(vec!["stake-addr-1".to_string()]));
        let mut sources = HashMap::new();
        sources.insert(
            model_id.to_string(),
            ModelSource { gguf_url: "http://gguf.example/model.gguf".to_string(), tokenizer_url: "http://gguf.example/tok".to_string(), tokenizer_sha256: "tok-hash".to_string() },
        );
        ShardEngine::new(csd, db, registry, stake_pool, ModelSourceConfig { sources }, None, None, expert_dispatcher)
    }

    fn test_manifest(model_id: &str) -> ModelManifest {
        ModelManifest { model_id: model_id.to_string(), arch_type: "MOE".to_string(), num_layers: 8, backbone_pom_root: "gguf-hash".to_string(), status: "ACTIVE".to_string() }
    }

    #[tokio::test]
    async fn moe_layer_range_fans_out_to_experts_and_feeds_combined_output_to_next_dense_shard() {
        let registry = Arc::new(MinerRegistry::new());

        // Registration order matters for `pick_next`'s round-robin — see
        // `MinerRegistry::pick_next`'s doc comment. Register the next-hop
        // dense miner FIRST (index 0), then the two expert-capable miners.
        let (tx_next, mut rx_next) = tokio::sync::mpsc::unbounded_channel();
        registry.register("next-hop-miner".to_string(), tx_next);
        let (tx_a, mut rx_a) = tokio::sync::mpsc::unbounded_channel();
        let (tx_b, mut rx_b) = tokio::sync::mpsc::unbounded_channel();
        registry.register("expert-a".to_string(), tx_a);
        registry.register("expert-b".to_string(), tx_b);
        registry.mark_expert_capable("expert-a", 24_000);
        registry.mark_expert_capable("expert-b", 24_000);

        let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let engine = Arc::new(test_engine(registry.clone(), "model-x", expert_dispatcher.clone()));

        // Force pick_next's round-robin to wrap around to index 0
        // ("next-hop-miner") for the next dense-shard dispatch below.
        *engine.last_assigned.lock() = Some("expert-b".to_string());

        let mut expert_groups = HashMap::new();
        expert_groups.insert((0u32, 4u32), vec![0u32, 1u32]);

        engine.pipelines.insert(
            "req-1".to_string(),
            PipelineState {
                shards: vec![dense(0, 0, 4), dense(1, 4, 8)],
                manifest: test_manifest("model-x"),
                prompt_hash: "ph".to_string(),
                max_tokens: 4,
                pos: 0,
                token_index: 0,
                current_input: ShardInputWire::Prompt { prompt_hex: "aa".to_string() },
                original_prompt_hex: "aa".to_string(),
                generated_token_ids: Vec::new(),
                contributions: std::collections::BTreeMap::new(),
                expert_groups,
                moe_layer_walk: None,
            },
        );
        engine.assignments.insert("req-1".to_string(), ShardAssignment { wallet: "primary-miner".to_string(), shard_index: 0, token_index: 0, moe_layer_step: None });

        let result = OpoiShardResult {
            request_id: "req-1".to_string(),
            shard_index: 0,
            token_index: 0,
            output_hash: "irrelevant-for-this-test".to_string(),
            output: ShardOutputWire { shape: vec![2], data_hex: f32_to_hex(&[1.0, 2.0]) },
            next_token_id: None,
            next_token_ids: None,
            // Single-layer range [0,4) -> exactly one outer entry, same
            // "range of length 1" degenerate case cs-miner's
            // route_only_over_range/route_only_at_layer proves agree.
            router_selection: Some(vec![vec![RouterExpertChoice { expert_id: 0, weight: 0.5 }, RouterExpertChoice { expert_id: 1, weight: 0.5 }]]),
            moe_layer_step: None,
        };

        let handle = tokio::spawn({
            let engine = engine.clone();
            async move { engine.handle_shard_result_inner("primary-miner", result).await }
        });

        // Both registered experts get exactly one opoi.expert_assign line —
        // proves the fan-out actually happened (part "b" of the task).
        let line_a = rx_a.recv().await.expect("expert-a got a line");
        let line_b = rx_b.recv().await.expect("expert-b got a line");
        assert!(line_a.contains("opoi.expert_assign"));
        assert!(line_b.contains("opoi.expert_assign"));

        for (expert_id, wallet, value) in [(0u32, "expert-a", 10.0f32), (1u32, "expert-b", 20.0f32)] {
            let reply = OpoiExpertResult {
                request_id: "req-1".to_string(),
                layer_start: 0,
                layer_end: 4,
                layer_idx: 3, // this single-layer range's only (boundary) layer
                expert_id,
                token_index: 0,
                output_hash: String::new(),
                output: ExpertOutputWire { shape: vec![2], data_hex: f32_to_hex(&[value, value]) },
            };
            expert_dispatcher.handle_expert_result(wallet, reply).await.expect("reply accepted");
        }

        handle.await.expect("task join").expect("handle_shard_result_inner ok");

        // Pipeline advanced past the EXPERT-graph shard, fed the
        // router-weighted-combined tensor forward as the input — part "c"
        // of the task: the SAME shape a plain dense-to-dense handoff uses.
        {
            let pipeline = engine.pipelines.get("req-1").expect("pipeline still in flight (not the last shard)");
            assert_eq!(pipeline.pos, 1);
            match &pipeline.current_input {
                ShardInputWire::Tensor { shape, data_hex } => {
                    assert_eq!(*shape, vec![2]);
                    let combined = hex_to_f32(data_hex).unwrap();
                    // 0.5*10 + 0.5*20 = 15.0 at both positions.
                    assert!((combined[0] - 15.0).abs() < 1e-5);
                    assert!((combined[1] - 15.0).abs() < 1e-5);
                }
                other => panic!("expected ShardInputWire::Tensor, got {other:?}"),
            }
        }

        // The NEXT dense shard was actually dispatched to "next-hop-miner".
        let next_line = rx_next.recv().await.expect("next-hop-miner got the next dense-shard assignment");
        assert!(next_line.contains("opoi.shard_assign"));
    }

    /// D2 multi-layer follow-up (2026-07-26, later session): proves this
    /// engine correctly consumes a genuine 2-layer-range `router_selection`
    /// (`Vec<Vec<RouterExpertChoice>>` with TWO entries, mirroring what
    /// cs-miner's `route_only_over_range` reports for a real 2-layer
    /// dense-boundary range) — validates AND accepts BOTH layers' entries
    /// (bounds-checked against `expert_ids`), but only turns the RANGE's
    /// LAST layer's selection into an actual `opoi.expert_assign` fan-out,
    /// exactly as `handle_expert_group_result`'s doc comment documents.
    /// The interior layer's (deliberately different, distinguishable)
    /// weights must NOT influence the combined output at all — proves this
    /// engine isn't accidentally averaging/blending across layers.
    #[tokio::test]
    async fn moe_multi_layer_range_dispatches_only_last_layer_selection() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx_next, mut rx_next) = tokio::sync::mpsc::unbounded_channel();
        registry.register("next-hop-miner".to_string(), tx_next);
        let (tx_a, mut rx_a) = tokio::sync::mpsc::unbounded_channel();
        let (tx_b, mut rx_b) = tokio::sync::mpsc::unbounded_channel();
        registry.register("expert-a".to_string(), tx_a);
        registry.register("expert-b".to_string(), tx_b);
        registry.mark_expert_capable("expert-a", 24_000);
        registry.mark_expert_capable("expert-b", 24_000);

        let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let engine = Arc::new(test_engine(registry.clone(), "model-x", expert_dispatcher.clone()));
        *engine.last_assigned.lock() = Some("expert-b".to_string());

        let mut expert_groups = HashMap::new();
        expert_groups.insert((0u32, 4u32), vec![0u32, 1u32]);

        engine.pipelines.insert(
            "req-multi".to_string(),
            PipelineState {
                shards: vec![dense(0, 0, 4), dense(1, 4, 8)],
                manifest: test_manifest("model-x"),
                prompt_hash: "ph".to_string(),
                max_tokens: 4,
                pos: 0,
                token_index: 0,
                current_input: ShardInputWire::Prompt { prompt_hex: "aa".to_string() },
                original_prompt_hex: "aa".to_string(),
                generated_token_ids: Vec::new(),
                contributions: std::collections::BTreeMap::new(),
                expert_groups,
                moe_layer_walk: None,
            },
        );
        engine.assignments.insert("req-multi".to_string(), ShardAssignment { wallet: "primary-miner".to_string(), shard_index: 0, token_index: 0, moe_layer_step: None });

        let result = OpoiShardResult {
            request_id: "req-multi".to_string(),
            shard_index: 0,
            token_index: 0,
            output_hash: "irrelevant-for-this-test".to_string(),
            output: ShardOutputWire { shape: vec![2], data_hex: f32_to_hex(&[1.0, 2.0]) },
            next_token_id: None,
            next_token_ids: None,
            // TWO layers in this range: offset 0 (the interior layer, real
            // data but NOT dispatched — deliberately lopsided weights, 0.9/0.1,
            // that would produce a very different combine than offset 1's
            // 0.5/0.5 if this engine ever mistakenly used them) and offset 1
            // (the boundary layer — the ONLY one actually dispatched below).
            router_selection: Some(vec![
                vec![RouterExpertChoice { expert_id: 0, weight: 0.9 }, RouterExpertChoice { expert_id: 1, weight: 0.1 }],
                vec![RouterExpertChoice { expert_id: 0, weight: 0.5 }, RouterExpertChoice { expert_id: 1, weight: 0.5 }],
            ]),
            moe_layer_step: None,
        };

        let handle = tokio::spawn({
            let engine = engine.clone();
            async move { engine.handle_shard_result_inner("primary-miner", result).await }
        });

        let line_a = rx_a.recv().await.expect("expert-a got a line");
        let line_b = rx_b.recv().await.expect("expert-b got a line");
        assert!(line_a.contains("opoi.expert_assign"));
        assert!(line_b.contains("opoi.expert_assign"));

        for (expert_id, wallet, value) in [(0u32, "expert-a", 10.0f32), (1u32, "expert-b", 20.0f32)] {
            let reply = OpoiExpertResult {
                request_id: "req-multi".to_string(),
                layer_start: 0,
                layer_end: 4,
                layer_idx: 3, // this 2-layer range's boundary layer (the OLD path's implicit `.last()`)
                expert_id,
                token_index: 0,
                output_hash: String::new(),
                output: ExpertOutputWire { shape: vec![2], data_hex: f32_to_hex(&[value, value]) },
            };
            expert_dispatcher.handle_expert_result(wallet, reply).await.expect("reply accepted");
        }

        handle.await.expect("task join").expect("handle_shard_result_inner ok");

        {
            let pipeline = engine.pipelines.get("req-multi").expect("pipeline still in flight (not the last shard)");
            assert_eq!(pipeline.pos, 1);
            match &pipeline.current_input {
                ShardInputWire::Tensor { shape, data_hex } => {
                    assert_eq!(*shape, vec![2]);
                    let combined = hex_to_f32(data_hex).unwrap();
                    // Must reflect the BOUNDARY layer's 0.5/0.5 weights
                    // (0.5*10 + 0.5*20 = 15.0), NOT the interior layer's
                    // 0.9/0.1 (which would give 0.9*10 + 0.1*20 = 11.0).
                    assert!((combined[0] - 15.0).abs() < 1e-5, "expected boundary-layer weights to be used, got {combined:?}");
                    assert!((combined[1] - 15.0).abs() < 1e-5, "expected boundary-layer weights to be used, got {combined:?}");
                }
                other => panic!("expected ShardInputWire::Tensor, got {other:?}"),
            }
        }

        let next_line = rx_next.recv().await.expect("next-hop-miner got the next dense-shard assignment");
        assert!(next_line.contains("opoi.shard_assign"));
    }

    /// The interior layer's `expert_id` gets bounds-checked too, even though
    /// it isn't dispatched — a malformed/dishonest interior entry must still
    /// drop the pipeline rather than be silently ignored just because it's
    /// not the layer this engine currently acts on.
    #[tokio::test]
    async fn moe_multi_layer_range_rejects_out_of_range_expert_id_even_in_an_interior_layer() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("primary-miner".to_string(), tx);

        let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let engine = test_engine(registry.clone(), "model-x", expert_dispatcher);

        let mut expert_groups = HashMap::new();
        expert_groups.insert((0u32, 4u32), vec![0u32, 1u32]);

        engine.pipelines.insert(
            "req-bad-interior".to_string(),
            PipelineState {
                shards: vec![dense(0, 0, 4), dense(1, 4, 8)],
                manifest: test_manifest("model-x"),
                prompt_hash: "ph".to_string(),
                max_tokens: 4,
                pos: 0,
                token_index: 0,
                current_input: ShardInputWire::Prompt { prompt_hex: "aa".to_string() },
                original_prompt_hex: "aa".to_string(),
                generated_token_ids: Vec::new(),
                contributions: std::collections::BTreeMap::new(),
                expert_groups,
                moe_layer_walk: None,
            },
        );
        engine.assignments.insert("req-bad-interior".to_string(), ShardAssignment { wallet: "primary-miner".to_string(), shard_index: 0, token_index: 0, moe_layer_step: None });

        let result = OpoiShardResult {
            request_id: "req-bad-interior".to_string(),
            shard_index: 0,
            token_index: 0,
            output_hash: "irrelevant-for-this-test".to_string(),
            output: ShardOutputWire { shape: vec![2], data_hex: f32_to_hex(&[1.0, 2.0]) },
            next_token_id: None,
            next_token_ids: None,
            router_selection: Some(vec![
                // Interior layer names expert_id 99 -- never registered for
                // this range.
                vec![RouterExpertChoice { expert_id: 99, weight: 1.0 }],
                vec![RouterExpertChoice { expert_id: 0, weight: 0.5 }, RouterExpertChoice { expert_id: 1, weight: 0.5 }],
            ]),
            moe_layer_step: None,
        };

        engine.handle_shard_result_inner("primary-miner", result).await.expect("dropping is not an error");

        assert!(engine.pipelines.get("req-bad-interior").is_none(), "pipeline must be dropped, not silently advanced");
    }

    /// TODO(D2-ROUTER-SELECTION) coverage: when the shard result carries no
    /// `router_selection` for a layer range that DOES have a registered
    /// EXPERT group, the pipeline must stall (not guess, not crash, not
    /// advance) — see this file's module doc and
    /// `handle_expert_group_result`'s doc comment.
    #[tokio::test]
    async fn missing_router_selection_stalls_the_pipeline_instead_of_guessing() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("primary-miner".to_string(), tx);

        let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let engine = test_engine(registry.clone(), "model-x", expert_dispatcher);

        let mut expert_groups = HashMap::new();
        expert_groups.insert((0u32, 4u32), vec![0u32, 1u32]);

        engine.pipelines.insert(
            "req-2".to_string(),
            PipelineState {
                shards: vec![dense(0, 0, 4), dense(1, 4, 8)],
                manifest: test_manifest("model-x"),
                prompt_hash: "ph".to_string(),
                max_tokens: 4,
                pos: 0,
                token_index: 0,
                current_input: ShardInputWire::Prompt { prompt_hex: "aa".to_string() },
                original_prompt_hex: "aa".to_string(),
                generated_token_ids: Vec::new(),
                contributions: std::collections::BTreeMap::new(),
                expert_groups,
                moe_layer_walk: None,
            },
        );
        engine.assignments.insert("req-2".to_string(), ShardAssignment { wallet: "primary-miner".to_string(), shard_index: 0, token_index: 0, moe_layer_step: None });

        let result = OpoiShardResult {
            request_id: "req-2".to_string(),
            shard_index: 0,
            token_index: 0,
            output_hash: "irrelevant-for-this-test".to_string(),
            output: ShardOutputWire { shape: vec![2], data_hex: f32_to_hex(&[1.0, 2.0]) },
            next_token_id: None,
            next_token_ids: None,
            router_selection: None, // the genuine, documented gap
            moe_layer_step: None,
        };

        engine.handle_shard_result_inner("primary-miner", result).await.expect("stalling is not an error");

        // Pipeline is still there, still parked at shard 0 — nothing
        // advanced, nothing was dispatched, nothing was fabricated.
        let pipeline = engine.pipelines.get("req-2").expect("pipeline was not dropped");
        assert_eq!(pipeline.pos, 0);
    }

    /// The narrower gap this file's module doc also documents: the LAST
    /// dense shard can have a registered EXPERT group too, but there is no
    /// wire mechanism yet to route a combined MoE output into final
    /// next-token decoding — the pipeline must fall back to the primary's
    /// own reported `next_token_id`, unchanged, exactly like a
    /// non-expert-routed model (no dispatch attempted for this case).
    #[tokio::test]
    async fn last_shard_with_expert_group_falls_back_to_primarys_next_token_id() {
        let registry = Arc::new(MinerRegistry::new());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        registry.register("primary-miner".to_string(), tx);

        let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let engine = test_engine(registry.clone(), "model-x", expert_dispatcher);

        // Single dense shard [0,8) — it IS the last (and only) shard —
        // with an EXPERT group registered at its own range.
        let mut expert_groups = HashMap::new();
        expert_groups.insert((0u32, 8u32), vec![0u32, 1u32]);

        engine.pipelines.insert(
            "req-3".to_string(),
            PipelineState {
                shards: vec![dense(0, 0, 8)],
                manifest: test_manifest("model-x"),
                prompt_hash: "ph".to_string(),
                max_tokens: 1, // 1 token: this result is both the last shard AND the last token
                pos: 0,
                token_index: 0,
                current_input: ShardInputWire::Prompt { prompt_hex: "aa".to_string() },
                original_prompt_hex: "aa".to_string(),
                generated_token_ids: Vec::new(),
                contributions: std::collections::BTreeMap::new(),
                expert_groups,
                moe_layer_walk: None,
            },
        );
        engine.assignments.insert("req-3".to_string(), ShardAssignment { wallet: "primary-miner".to_string(), shard_index: 0, token_index: 0, moe_layer_step: None });

        let result = OpoiShardResult {
            request_id: "req-3".to_string(),
            shard_index: 0,
            token_index: 0,
            output_hash: "irrelevant-for-this-test".to_string(),
            output: ShardOutputWire { shape: vec![2], data_hex: f32_to_hex(&[1.0, 2.0]) },
            next_token_id: Some(42),
            next_token_ids: None,
            router_selection: None, // no expert dispatch should even be attempted here
            moe_layer_step: None,
        };

        // is_last_token => true (max_tokens=1) => `finalize_pipeline` runs
        // and removes the request from `csd`/`db`... but `submit_shard_result`
        // requires the `stake_pool`; with an empty stake pool `try_each`
        // fails, so the pipeline stays put rather than finalizing — either
        // way the assertion below (no expert dispatch attempted, no panic)
        // holds regardless of which of those two paths this takes.
        engine.handle_shard_result_inner("primary-miner", result).await.expect("must not error");
    }

    /// D2 multi-layer STATEFUL resume (2026-07-26 follow-up session) — the
    /// actual new end-to-end behavior this session's redesign delivers:
    /// a genuine 2-layer range [0,4) walked ONE LAYER AT A TIME, both
    /// layers' experts REALLY dispatched externally (not just the
    /// boundary's), chained correctly, driven through the REAL
    /// `dispatch_current` (not hand-constructed assignments) so the
    /// wallet-pinning logic itself is exercised, not just
    /// `handle_shard_result_inner`'s reaction to a result.
    ///
    /// Proves, in order:
    ///  1. The FIRST dispatch for a multi-layer MoE range sets
    ///     `moe_layer_step: Some(layer_start)` and pins the chosen wallet.
    ///  2. The interior layer's (layer 0) real dispatch/combine does NOT
    ///     advance `pos` and does NOT touch on-chain submission — only
    ///     updates `current_input`/`moe_layer_walk` and re-dispatches.
    ///  3. The RESUME dispatch for layer 1 goes back to the EXACT SAME
    ///     wallet (never round-robins) and asks for `moe_layer_step: 1`.
    ///  4. The boundary layer's (layer 1) combine DOES advance `pos` and
    ///     clears `moe_layer_walk`, using the BOUNDARY's own combined
    ///     weights (deliberately different from the interior layer's, so a
    ///     bug that confused the two would be caught).
    ///  5. The NEXT dense shard (no expert group) is dispatched normally,
    ///     round-robining away from the pinned wallet — proving the pin
    ///     doesn't leak into unrelated shards.
    #[tokio::test]
    async fn moe_multi_layer_stateful_walk_dispatches_every_layer_pinned_to_the_same_wallet() {
        let registry = Arc::new(MinerRegistry::new());

        // Registration order matters for `pick_next`'s round-robin:
        // "primary-miner" first (picked for shard 0's fresh dispatch, with
        // `last_assigned` starting `None`), then "next-hop-miner" (picked
        // next, for shard 1, once `last_assigned` has advanced past
        // "primary-miner").
        let (tx_primary, mut rx_primary) = tokio::sync::mpsc::unbounded_channel();
        registry.register("primary-miner".to_string(), tx_primary);
        let (tx_next, mut rx_next) = tokio::sync::mpsc::unbounded_channel();
        registry.register("next-hop-miner".to_string(), tx_next);
        let (tx_a, mut rx_a) = tokio::sync::mpsc::unbounded_channel();
        let (tx_b, mut rx_b) = tokio::sync::mpsc::unbounded_channel();
        registry.register("expert-a".to_string(), tx_a);
        registry.register("expert-b".to_string(), tx_b);
        registry.mark_expert_capable("expert-a", 24_000);
        registry.mark_expert_capable("expert-b", 24_000);

        let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let engine = Arc::new(test_engine(registry.clone(), "model-x", expert_dispatcher.clone()));

        // A genuine 2-LAYER range [0,2) — layer 0 is the interior layer,
        // layer 1 is the boundary. (Earlier D2 tests in this module use
        // [0,4) but only ever dispatch ITS boundary layer 3 in one shot;
        // here we need EXACTLY two absolute layers, 0 and 1, to walk.)
        let mut expert_groups = HashMap::new();
        expert_groups.insert((0u32, 2u32), vec![0u32, 1u32]);

        engine.pipelines.insert(
            "req-walk".to_string(),
            PipelineState {
                shards: vec![dense(0, 0, 2), dense(1, 2, 8)],
                manifest: test_manifest("model-x"),
                prompt_hash: "ph".to_string(),
                max_tokens: 4,
                pos: 0,
                token_index: 0,
                current_input: ShardInputWire::Prompt { prompt_hex: "aa".to_string() },
                original_prompt_hex: "aa".to_string(),
                generated_token_ids: Vec::new(),
                contributions: std::collections::BTreeMap::new(),
                expert_groups,
                moe_layer_walk: None,
            },
        );

        let source = ModelSource {
            gguf_url: "http://gguf.example/model.gguf".to_string(),
            tokenizer_url: "http://gguf.example/tok".to_string(),
            tokenizer_sha256: "tok-hash".to_string(),
        };

        // ---- Step 1: fresh dispatch for shard 0 (2-layer range [0,4)) -------
        engine.dispatch_current("req-walk", &source).await;
        let assign_line_1 = rx_primary.recv().await.expect("primary-miner got the fresh dispatch");
        assert!(assign_line_1.contains(r#""moe_layer_step":0"#), "fresh dispatch should ask for layer_idx=layer_start=0: {assign_line_1}");

        {
            let pipeline = engine.pipelines.get("req-walk").unwrap();
            assert_eq!(pipeline.moe_layer_walk, Some((0, "primary-miner".to_string())));
        }

        // ---- Miner replies with layer 0's (INTERIOR) real selection ---------
        let result1 = OpoiShardResult {
            request_id: "req-walk".to_string(),
            shard_index: 0,
            token_index: 0,
            output_hash: "irrelevant-for-this-test".to_string(),
            output: ShardOutputWire { shape: vec![2], data_hex: f32_to_hex(&[1.0, 2.0]) },
            next_token_id: None,
            next_token_ids: None,
            router_selection: Some(vec![vec![RouterExpertChoice { expert_id: 0, weight: 0.9 }, RouterExpertChoice { expert_id: 1, weight: 0.1 }]]),
            moe_layer_step: Some(0),
        };
        let handle1 = tokio::spawn({
            let engine = engine.clone();
            async move { engine.handle_shard_result_inner("primary-miner", result1).await }
        });
        let line_a1 = rx_a.recv().await.expect("expert-a got a line for layer 0");
        let line_b1 = rx_b.recv().await.expect("expert-b got a line for layer 0");
        assert!(line_a1.contains("opoi.expert_assign"));
        assert!(line_b1.contains("opoi.expert_assign"));
        for (expert_id, wallet, value) in [(0u32, "expert-a", 10.0f32), (1u32, "expert-b", 20.0f32)] {
            let reply = OpoiExpertResult {
                request_id: "req-walk".to_string(),
                layer_start: 0,
                layer_end: 2,
                layer_idx: 0,
                expert_id,
                token_index: 0,
                output_hash: String::new(),
                output: ExpertOutputWire { shape: vec![2], data_hex: f32_to_hex(&[value, value]) },
            };
            expert_dispatcher.handle_expert_result(wallet, reply).await.expect("reply accepted");
        }
        handle1.await.expect("task join").expect("handle_shard_result_inner ok (interior)");

        // Interior layer must NOT advance pos, NOT clear the pin, and must
        // feed the REAL combined output (0.9*10 + 0.1*20 = 11.0) forward as
        // the resume step's input.
        {
            let pipeline = engine.pipelines.get("req-walk").unwrap();
            assert_eq!(pipeline.pos, 0, "interior layer must not advance pos");
            assert_eq!(pipeline.moe_layer_walk, Some((1, "primary-miner".to_string())), "walk must advance to layer 1, still pinned to primary-miner");
            match &pipeline.current_input {
                ShardInputWire::Tensor { data_hex, .. } => {
                    let vals = hex_to_f32(data_hex).unwrap();
                    assert!((vals[0] - 11.0).abs() < 1e-5, "expected interior combine 0.9*10+0.1*20=11.0, got {vals:?}");
                }
                other => panic!("expected Tensor input after interior combine, got {other:?}"),
            }
        }

        // ---- The RESUME dispatch for layer 1 must go back to the SAME wallet
        let assign_line_2 = rx_primary.recv().await.expect("primary-miner (same wallet) got the resume dispatch too");
        assert!(assign_line_2.contains(r#""moe_layer_step":1"#), "resume dispatch should ask for layer_idx=1: {assign_line_2}");

        // ---- Miner replies with layer 1's (BOUNDARY) real selection ---------
        let result2 = OpoiShardResult {
            request_id: "req-walk".to_string(),
            shard_index: 0,
            token_index: 0,
            output_hash: "boundary-hash".to_string(),
            output: ShardOutputWire { shape: vec![2], data_hex: f32_to_hex(&[3.0, 4.0]) },
            next_token_id: None,
            next_token_ids: None,
            router_selection: Some(vec![vec![RouterExpertChoice { expert_id: 0, weight: 0.5 }, RouterExpertChoice { expert_id: 1, weight: 0.5 }]]),
            moe_layer_step: Some(1),
        };
        let handle2 = tokio::spawn({
            let engine = engine.clone();
            async move { engine.handle_shard_result_inner("primary-miner", result2).await }
        });
        let line_a2 = rx_a.recv().await.expect("expert-a got a line for layer 1 (boundary)");
        let line_b2 = rx_b.recv().await.expect("expert-b got a line for layer 1 (boundary)");
        assert!(line_a2.contains("opoi.expert_assign"));
        assert!(line_b2.contains("opoi.expert_assign"));
        for (expert_id, wallet, value) in [(0u32, "expert-a", 100.0f32), (1u32, "expert-b", 200.0f32)] {
            let reply = OpoiExpertResult {
                request_id: "req-walk".to_string(),
                layer_start: 0,
                layer_end: 2,
                layer_idx: 1,
                expert_id,
                token_index: 0,
                output_hash: String::new(),
                output: ExpertOutputWire { shape: vec![2], data_hex: f32_to_hex(&[value, value]) },
            };
            expert_dispatcher.handle_expert_result(wallet, reply).await.expect("reply accepted");
        }
        handle2.await.expect("task join").expect("handle_shard_result_inner ok (boundary)");

        // Boundary layer MUST advance pos and clear the pin, using the
        // BOUNDARY's own combined weights (0.5*100 + 0.5*200 = 150.0) — NOT
        // the interior layer's 11.0, proving the two steps' data never
        // cross-contaminate.
        {
            let pipeline = engine.pipelines.get("req-walk").expect("pipeline still in flight (not the last shard)");
            assert_eq!(pipeline.pos, 1);
            assert_eq!(pipeline.moe_layer_walk, None, "walk must be cleared once the boundary layer completes");
            match &pipeline.current_input {
                ShardInputWire::Tensor { data_hex, .. } => {
                    let vals = hex_to_f32(data_hex).unwrap();
                    assert!((vals[0] - 150.0).abs() < 1e-5, "expected boundary combine 0.5*100+0.5*200=150.0, got {vals:?}");
                }
                other => panic!("expected Tensor input after boundary combine, got {other:?}"),
            }
        }

        // Next dense shard (shard 1, no expert group) dispatched normally —
        // round-robins AWAY from the pinned wallet, plain (no
        // moe_layer_step), proving the pin never leaks into an unrelated
        // shard.
        let next_line = rx_next.recv().await.expect("next-hop-miner got the next dense-shard assignment");
        assert!(next_line.contains("opoi.shard_assign"));
        assert!(!next_line.contains("moe_layer_step"));
    }

    /// D2 multi-layer STATEFUL resume: if the wallet pinned mid-walk
    /// disconnects, that range-walk's residual/KV state is gone for good
    /// (it only ever existed on that one machine — see
    /// `PipelineState::moe_layer_walk`'s doc) — `dispatch_current` must
    /// drop the whole pipeline rather than silently resuming against some
    /// OTHER connected miner, which would compute a wrong result (a fresh
    /// miner has no residual to finish the interior layer's combine with).
    #[tokio::test]
    async fn dispatch_current_drops_pipeline_when_pinned_wallet_for_a_mid_walk_resume_is_gone() {
        let registry = Arc::new(MinerRegistry::new());
        // "ghost-miner" (the pinned wallet) is deliberately NEVER registered
        // — simulates it having already disconnected.

        let expert_dispatcher = Arc::new(ExpertDispatcher::new(registry.clone()));
        let engine = Arc::new(test_engine(registry.clone(), "model-x", expert_dispatcher));

        let mut expert_groups = HashMap::new();
        expert_groups.insert((0u32, 2u32), vec![0u32, 1u32]);

        engine.pipelines.insert(
            "req-disc".to_string(),
            PipelineState {
                shards: vec![dense(0, 0, 2), dense(1, 2, 8)],
                manifest: test_manifest("model-x"),
                prompt_hash: "ph".to_string(),
                max_tokens: 4,
                pos: 0,
                token_index: 0,
                current_input: ShardInputWire::Tensor { shape: vec![2], data_hex: f32_to_hex(&[1.0, 2.0]) },
                original_prompt_hex: "aa".to_string(),
                generated_token_ids: Vec::new(),
                contributions: std::collections::BTreeMap::new(),
                expert_groups,
                // Mid-walk state: pinned to a wallet that is NOT connected.
                moe_layer_walk: Some((1, "ghost-miner".to_string())),
            },
        );

        let source = ModelSource {
            gguf_url: "http://gguf.example/model.gguf".to_string(),
            tokenizer_url: "http://gguf.example/tok".to_string(),
            tokenizer_sha256: "tok-hash".to_string(),
        };

        engine.dispatch_current("req-disc", &source).await;

        assert!(
            engine.pipelines.get("req-disc").is_none(),
            "pipeline must be dropped when the pinned wallet is gone, not silently resumed against a different miner"
        );
    }
}
