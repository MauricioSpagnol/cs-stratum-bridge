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
//! MoE is explicitly out of scope here: any model whose Model Execution
//! Graph contains an `EXPERT` shard is skipped entirely (logged, not
//! attempted) — giant MoE models are served single-node via the offload
//! path instead (F9-G/F15-M, `CS COIN OPoI MELHOR IMPLEMENTAÇÃO.txt`
//! Sprint 4/8).
//!
//! Scope note (Sessão 3, deliberate): pipeline state below is in-memory
//! only (`DashMap`), not persisted — a bridge restart loses in-flight shard
//! pipelines (they'll simply be re-picked-up from PENDING on the next poll
//! tick and restarted from token_index 0, same as a fresh request). Adding
//! restart-safe persistence for shard pipelines is future work, not a
//! correctness requirement for this first working version.

use std::collections::BTreeMap;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use sqlx::PgPool;

use crate::db;
use crate::miner_registry::MinerRegistry;
use crate::opoi::b3lite::{self, B3LiteConfig};
use crate::opoi::engine::do_commit;
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
}

impl ShardEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        csd: Arc<CsdRpcClient>, db: PgPool, registry: Arc<MinerRegistry>, stake_pool: Arc<StakePool>, model_sources: ModelSourceConfig,
        speculative: Option<Arc<SpeculativeEngine>>, b3lite: Option<B3LiteConfig>,
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
            if graph.shards.iter().any(|s| s.shard_type == "EXPERT") {
                tracing::info!(model = %req.model, request_id = %req.request_id, "MoE model — shard pipeline not implemented yet, skipping (see F9-G/F15-M for the single-node path)");
                continue;
            }
            let Some(source) = self.model_sources.get(&req.model) else {
                tracing::warn!(model = %req.model, "no MODEL_SOURCES_JSON entry — cannot resolve a download URL, skipping");
                continue;
            };

            let mut shards = graph.shards.clone();
            shards.sort_by_key(|s| s.shard_index);

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

    async fn dispatch_current(&self, request_id: &str, source: &ModelSource) {
        let Some(pipeline) = self.pipelines.get(request_id) else { return };
        let shard = &pipeline.shards[pipeline.pos];
        let is_entry = pipeline.pos == 0;

        let wallet = {
            let mut last = self.last_assigned.lock();
            let Some(wallet) = self.registry.pick_next(&last) else {
                tracing::debug!(request_id = %request_id, "no miners connected right now, will retry dispatching this shard next tick");
                return;
            };
            *last = Some(wallet.clone());
            wallet
        };
        let Some(tx) = self.registry.get(&wallet) else { return };

        let assign = OpoiShardAssign {
            request_id: request_id.to_string(),
            model_id: pipeline.manifest.model_id.clone(),
            shard_index: shard.shard_index,
            layer_start: shard.layer_start.unwrap_or(0),
            layer_end: shard.layer_end.unwrap_or(0),
            total_layers: pipeline.manifest.num_layers,
            token_index: pipeline.token_index,
            max_tokens: pipeline.max_tokens,
            input: pipeline.current_input.clone(),
            model_gguf_url: source.gguf_url.clone(),
            model_gguf_sha256: pipeline.manifest.backbone_pom_root.clone(),
            model_tokenizer_url: is_entry.then(|| source.tokenizer_url.clone()),
            model_tokenizer_sha256: is_entry.then(|| source.tokenizer_sha256.clone()),
        };
        self.assignments.insert(
            request_id.to_string(),
            ShardAssignment { wallet: wallet.clone(), shard_index: shard.shard_index, token_index: pipeline.token_index },
        );
        let _ = tx.send(build_shard_assign_line(&assign));
        tracing::info!(request_id = %request_id, shard_index = shard.shard_index, token_index = pipeline.token_index, wallet = %wallet, "dispatched shard");
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
            if assignment.wallet != wallet || assignment.shard_index != result.shard_index || assignment.token_index != result.token_index {
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
        // the pipeline's last step.
        if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
            *pipeline.contributions.entry(wallet.to_string()).or_insert(0) += 1;
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

        if !is_last_shard {
            // Advance to the next shard in the SAME token step, feeding it
            // this shard's real output tensor.
            if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
                pipeline.pos += 1;
                pipeline.current_input = ShardInputWire::Tensor { shape: result.output.shape.clone(), data_hex: result.output.data_hex.clone() };
            }
            self.dispatch_current(&request_id, &source).await;
            return Ok(());
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
        }
        self.dispatch_current(&request_id, &source).await;
        Ok(())
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
}
