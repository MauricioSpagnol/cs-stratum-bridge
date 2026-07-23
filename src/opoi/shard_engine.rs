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

use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use crate::miner_registry::MinerRegistry;
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
    max_tokens: u32,
    /// Index into `shards` (NOT `shard_index` directly, though they
    /// coincide for a well-formed dense-only graph starting at 0).
    pos: usize,
    token_index: u32,
    /// What the currently-dispatched shard was given as input — kept so a
    /// re-dispatch (e.g. after a miner disconnects mid-shard) can resend
    /// the identical assignment instead of needing to reconstruct it.
    current_input: ShardInputWire,
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
}

impl ShardEngine {
    pub fn new(
        csd: Arc<CsdRpcClient>, registry: Arc<MinerRegistry>, stake_pool: Arc<StakePool>, model_sources: ModelSourceConfig,
        speculative: Option<Arc<SpeculativeEngine>>,
    ) -> Self {
        Self {
            csd,
            registry,
            stake_pool,
            model_sources,
            pipelines: DashMap::new(),
            assignments: DashMap::new(),
            last_assigned: Mutex::new(None),
            prompt_cache: DashMap::new(),
            speculative,
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
                    spec.start_pipeline(req.request_id.clone(), manifest, shards, req.max_tokens, prompt_hex.clone(), source).await;
                    self.prompt_cache.remove(&req.request_id);
                    continue;
                }
            }

            self.pipelines.insert(
                req.request_id.clone(),
                PipelineState {
                    shards,
                    manifest,
                    max_tokens: req.max_tokens,
                    pos: 0,
                    token_index: 0,
                    current_input: ShardInputWire::Prompt { prompt_hex: prompt_hex.clone() },
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

        if is_last_token {
            tracing::info!(request_id = %request_id, "shard pipeline finished (max_tokens reached)");
            self.pipelines.remove(&request_id);
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

/// Only used by tests today (hashing isn't otherwise this module's job —
/// `shard_compute.rs` on the miner side computes the real hash). Kept here
/// so a future on-bridge sanity re-hash (defense in depth) has an obvious
/// home if it's ever added.
#[allow(dead_code)]
fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}
