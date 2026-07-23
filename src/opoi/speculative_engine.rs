//! F15-L (speculative decoding, Revisão v6): the speculative-decoding
//! counterpart of `shard_engine.rs` — dispatches a draft-capable miner's
//! local proposals and a batched multi-token verify through the SAME dense
//! shard pipeline `shard_engine.rs` already drives one token at a time, then
//! runs greedy rejection sampling (`accept the drafted tokens that match the
//! target's own predictions, correct at the first mismatch`) and rolls every
//! session's KV-cache back to the true committed length before starting the
//! next round.
//!
//! NOT an extension of `ShardEngine` (nor of `OpoiEngine`) — same reasoning
//! as `shard_engine.rs`'s own module doc for why IT isn't an extension of
//! `OpoiEngine`: the state machine here (draft dispatch, a MULTI-token
//! verify batch relayed through every shard, greedy-accept/reject
//! bookkeeping, per-round KV rollback across every shard AND the draft) is
//! different enough from ShardEngine's single-token relay loop that forcing
//! them into one struct would just mean two unrelated jobs tangled
//! together. `shard_engine.rs`'s `poll_and_start_tick` delegates HERE
//! instead of creating its own `PipelineState` whenever a draft model is
//! configured for the request's model AND a draft-capable miner is
//! currently connected (see `ShardEngine::is_speculative_eligible` and its
//! call site) — falling back to `ShardEngine`'s existing plain per-token
//! dispatch otherwise, which stays completely unchanged.
//!
//! ## The round algorithm (implemented exactly as derived — see
//! `cs-miner/src/opoi/speculative_round.rs`'s module doc for the full
//! derivation, including one bookkeeping correction found while deriving it:
//! the design doc's own `keep_until_pos = base + accepted_count` formula is
//! one short whenever a carry token is present, because carrying a token
//! into a round and verifying it via that round's batch IS a real physical
//! KV feed, exactly like any other verified-correct drafted token — this
//! module's `kv_keep_len` folds that back in, ported byte-for-byte from the
//! cs-miner module above (separate crates/repos, can't literally share the
//! function — see this project's existing wire-type mirroring convention).
//!
//! ## Deliberate deviation from the literal round-0-draft-seed and
//! finalize-scope wording elsewhere in the design doc (documented here since
//! this is the one file that had to resolve it): round 0's draft-generate
//! call seeds with `DraftInputWire::Prompt` (not `NextTokenId`) — the ONLY
//! choice that keeps the draft's own first candidate aligned with the SAME
//! KV position the target's entry shard is about to verify (position
//! `prompt_len`, i.e. the position `next_token_id` from prompt-processing
//! already predicts). And every REAL shard (not only the pipeline's last
//! one) finalizes on the round that reaches `max_tokens` — mirroring
//! `shard_engine.rs`'s own established behavior of submitting on-chain for
//! whichever shard just completed its own last-token step, so every shard's
//! work still gets its own on-chain commitment; only the DRAFT's session
//! never finalizes (`finalize: false` always — a draft session has no
//! on-chain meaning).
//!
//! ## Known, deliberately out-of-scope limitations (see also the acceptance
//! test's module doc in cs-miner's `examples/speculative_decoding_test.rs`):
//! - The draft's own local KV-cache session can drift by exactly one token
//!   after a FULLY-accepted round (a structural gap in any "generate k
//!   candidates by chaining" loop — the last candidate is, by construction,
//!   never itself fed back in). This can never corrupt the final committed
//!   sequence (only this module's OWN bookkeeping — verified independently
//!   of the draft — decides what's committed) but can lower future
//!   acceptance rate. Fixing it needs a new `DraftInputWire` variant on
//!   cs-miner's side (e.g. `Resync{token_id}`); out of scope here since that
//!   wire contract is already built and treated as fixed this session.
//! - If the draft-capable miner a pipeline is using disconnects mid-flight,
//!   its local session (and therefore that pipeline's ability to keep
//!   speculating) is unrecoverable — a NEW draft-capable miner has no
//!   history for this request. `on_disconnect` drops the whole pipeline in
//!   that case (logged) rather than attempting to reseed a fresh draft
//!   session from scratch. `ShardEngine`'s own non-speculative path has no
//!   such gap (a fresh shard session only ever needs the prompt, which is
//!   always available) — this is a genuine speculative-only limitation.
//! - No restart-recovery for an in-flight rollback round (mirrors
//!   `shard_engine.rs`'s own documented lack of restart-safe persistence —
//!   Sessão 3 scope note in that file).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex;

use crate::miner_registry::MinerRegistry;
use crate::opoi::shard_engine::ModelSource;
use crate::opoi::wire::{
    build_draft_generate_line, build_kv_rollback_line, build_shard_assign_line, DraftInputWire, OpoiDraftGenerate, OpoiDraftResult,
    OpoiKvRollback, OpoiKvRollbackAck, OpoiShardAssign, OpoiShardResult, ShardInputWire, DRAFT_SESSION_SHARD_INDEX,
};
use crate::rpc::types::{ModelManifest, ShardDescriptor};
use crate::rpc::CsdRpcClient;
use crate::stake_pool::StakePool;

// ── speculative_round.rs, ported byte-for-byte from cs-miner's copy ──────
// (see this module's doc comment for why it's a port, not a shared crate).

/// Outcome of verifying one speculative round's batch — mirrors
/// `cs-miner/src/opoi/speculative_round.rs::RoundOutcome`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RoundOutcome {
    accepted_count: usize,
    settle_token: u32,
    full_accept: bool,
}

fn greedy_verify_round(carry: Option<u32>, expected0: u32, draft_token_ids: &[u32], target_logits: &[u32]) -> RoundOutcome {
    let carry_len = if carry.is_some() { 1 } else { 0 };
    assert_eq!(target_logits.len(), draft_token_ids.len() + carry_len, "target_logits must have exactly one entry per verify-batch position");

    let mut expected = expected0;
    let mut accepted_count = 0usize;
    for (i, &logit) in target_logits.iter().enumerate() {
        if i == 0 && carry.is_some() {
            expected = logit;
            continue;
        }
        let d = draft_token_ids[i - carry_len];
        if d == expected {
            accepted_count += 1;
            expected = logit;
        } else {
            return RoundOutcome { accepted_count, settle_token: expected, full_accept: false };
        }
    }
    RoundOutcome { accepted_count, settle_token: expected, full_accept: true }
}

fn kv_keep_len(base_start: usize, carry: Option<u32>, accepted_count: usize) -> usize {
    base_start + if carry.is_some() { 1 } else { 0 } + accepted_count
}

fn next_k(current_k: u32, accepted_count: usize, k_this_round: usize) -> u32 {
    const MIN_K: u32 = 2;
    const MAX_K: u32 = 8;
    if k_this_round == 0 {
        return current_k.clamp(MIN_K, MAX_K);
    }
    let rate = accepted_count as f64 / k_this_round as f64;
    let next = if rate > 0.7 { current_k + 1 } else if rate < 0.5 { current_k.saturating_sub(1) } else { current_k };
    next.clamp(MIN_K, MAX_K)
}

// ── draft model source config (analogous to shard_engine::ModelSourceConfig) ──

#[derive(Clone, serde::Deserialize)]
pub struct DraftModelSource {
    pub draft_model_id: String,
    pub draft_gguf_url: String,
    pub draft_gguf_sha256: String,
    pub draft_tokenizer_url: String,
    pub draft_tokenizer_sha256: String,
    pub draft_total_layers: u32,
}

/// Loaded once from `DRAFT_MODEL_SOURCES_JSON` (env var: a JSON object
/// `{"TARGET_MODEL_ID": {"draft_model_id": ..., ...}}`). A target model with
/// no entry here is never speculatively dispatched — `ShardEngine` falls
/// back to its existing plain per-token path for it, unconditionally.
pub struct DraftModelConfig {
    sources: HashMap<String, DraftModelSource>,
}

impl DraftModelConfig {
    pub fn from_env() -> Self {
        let raw = std::env::var("DRAFT_MODEL_SOURCES_JSON").unwrap_or_default();
        let sources = if raw.trim().is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&raw).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "DRAFT_MODEL_SOURCES_JSON is set but failed to parse; speculative decoding disabled for all models");
                HashMap::new()
            })
        };
        Self { sources }
    }

    pub fn get(&self, target_model_id: &str) -> Option<&DraftModelSource> {
        self.sources.get(target_model_id)
    }
}

// ── pipeline state machine ────────────────────────────────────────────────

enum Phase {
    /// The initial prompt-processing pass through the target pipeline —
    /// identical in spirit to `shard_engine.rs`'s own token_index==0 step,
    /// not itself a "speculative round". `project_all` is always `false`
    /// here (single next-token prediction, same as the non-speculative
    /// path's first step).
    PromptProcessing { pos: usize, current_input: ShardInputWire },
    /// Waiting for `opoi.draft_result` from the pinned draft-capable miner.
    AwaitingDraft,
    /// Relaying this round's verify batch through every shard in order.
    Verifying { pos: usize, current_input: ShardInputWire, draft_token_ids: Vec<u32> },
    /// Waiting for every shard's (and the draft's) `opoi.kv_rollback_ack`
    /// before starting the next round or finishing.
    AwaitingRollback(RollbackState),
}

struct RollbackState {
    /// shard_index values still awaiting an ack — includes
    /// `DRAFT_SESSION_SHARD_INDEX` for the draft's own session.
    pending: HashSet<u32>,
    accepted_count: usize,
    k_this_round: usize,
    settle_token: u32,
    keep_len: usize,
    is_finalizing_round: bool,
}

struct SpecPipeline {
    shards: Vec<ShardDescriptor>,
    manifest: ModelManifest,
    source: ModelSource,
    draft_source: DraftModelSource,
    max_tokens: u32,
    /// Stashed for round 0's draft-generate seed (`DraftInputWire::Prompt`)
    /// — by the time `Phase` has moved on to `AwaitingDraft`, the original
    /// `ShardInputWire::Prompt` it came from is gone.
    prompt_hex: String,
    /// Pinned once at round 0 — the draft's own KV-cache session lives on
    /// THIS specific miner process; a different draft-capable miner has no
    /// history for this request (see module doc's disconnect-limitation
    /// note).
    draft_wallet: Option<String>,
    base: usize,
    carry: Option<u32>,
    expected0: u32,
    k: u32,
    round: u32,
    phase: Phase,
}

#[derive(Default)]
struct ShardStepAssignment {
    wallet: String,
    shard_index: u32,
}

pub struct SpeculativeEngine {
    csd: Arc<CsdRpcClient>,
    registry: Arc<MinerRegistry>,
    stake_pool: Arc<StakePool>,
    draft_sources: DraftModelConfig,
    tokenizer_cache_dir: PathBuf,
    pipelines: DashMap<String, SpecPipeline>,
    /// In-flight shard-relay assignment (PromptProcessing or Verifying
    /// phase) — same one-in-flight-per-request invariant as
    /// `shard_engine::ShardEngine`'s own `assignments` map.
    shard_assignments: DashMap<String, ShardStepAssignment>,
    /// In-flight draft-generate assignment (request_id -> wallet).
    draft_assignments: DashMap<String, String>,
    last_shard_wallet: Mutex<Option<String>>,
}

impl SpeculativeEngine {
    pub fn new(csd: Arc<CsdRpcClient>, registry: Arc<MinerRegistry>, stake_pool: Arc<StakePool>, draft_sources: DraftModelConfig) -> Self {
        Self {
            csd,
            registry,
            stake_pool,
            draft_sources,
            tokenizer_cache_dir: PathBuf::from("./tokenizer_cache"),
            pipelines: DashMap::new(),
            shard_assignments: DashMap::new(),
            draft_assignments: DashMap::new(),
            last_shard_wallet: Mutex::new(None),
        }
    }

    /// The two conditions `shard_engine.rs`'s `poll_and_start_tick` checks
    /// before delegating here instead of starting its own plain
    /// `PipelineState`.
    pub fn is_eligible(&self, target_model_id: &str) -> bool {
        self.draft_sources.get(target_model_id).is_some() && self.registry.pick_draft_capable(&None).is_some()
    }

    /// `shard_engine.rs`'s `handle_shard_result_inner` checks this to
    /// decide whether an incoming `opoi.shard_result` belongs to THIS
    /// engine's own state machine instead of its plain per-token one — a
    /// request is dispatched via exactly one of the two for its whole
    /// lifetime, decided once in `poll_and_start_tick`.
    pub fn has_pipeline(&self, request_id: &str) -> bool {
        self.pipelines.contains_key(request_id)
    }

    /// Starts a new speculative pipeline — called by `ShardEngine` in place
    /// of its own pipeline-creation, once `is_eligible` and every other
    /// non-speculative precondition (coordinator claim, dense-only graph,
    /// etc.) already passed there.
    pub async fn start_pipeline(
        &self, request_id: String, manifest: ModelManifest, mut shards: Vec<ShardDescriptor>, max_tokens: u32, prompt_hex: String, source: ModelSource,
    ) {
        let Some(draft_source) = self.draft_sources.get(&manifest.model_id).cloned() else {
            tracing::warn!(request_id = %request_id, "SpeculativeEngine::start_pipeline called with no draft source configured — should not happen");
            return;
        };
        shards.sort_by_key(|s| s.shard_index);

        self.pipelines.insert(
            request_id.clone(),
            SpecPipeline {
                shards,
                manifest,
                source,
                draft_source,
                max_tokens,
                prompt_hex: prompt_hex.clone(),
                draft_wallet: None,
                base: 0,
                carry: None,
                expected0: 0,
                k: 4,
                round: 0,
                phase: Phase::PromptProcessing { pos: 0, current_input: ShardInputWire::Prompt { prompt_hex } },
            },
        );
        self.dispatch_shard_step(&request_id).await;
    }

    /// Mirrors `ShardEngine::retry_stalled_pipelines` — re-dispatches
    /// whatever a pipeline is currently waiting on if nothing is actually
    /// in flight for it (first dispatch found no miner connected, or the
    /// assigned miner disconnected). Call this on the same poll tick as
    /// `ShardEngine::poll_and_start_tick`.
    pub async fn retry_stalled_tick(&self) {
        let stalled_shard: Vec<String> = self
            .pipelines
            .iter()
            .filter(|e| matches!(e.phase, Phase::PromptProcessing { .. } | Phase::Verifying { .. }) && !self.shard_assignments.contains_key(e.key()))
            .map(|e| e.key().clone())
            .collect();
        for request_id in stalled_shard {
            self.dispatch_shard_step(&request_id).await;
        }

        let stalled_draft: Vec<String> = self
            .pipelines
            .iter()
            .filter(|e| matches!(e.phase, Phase::AwaitingDraft) && !self.draft_assignments.contains_key(e.key()))
            .map(|e| e.key().clone())
            .collect();
        for request_id in stalled_draft {
            self.dispatch_draft(&request_id).await;
        }
    }

    // ── shard relay (prompt-processing pass, and each round's verify pass) ──

    async fn dispatch_shard_step(&self, request_id: &str) {
        let Some(pipeline) = self.pipelines.get(request_id) else { return };
        let (pos, current_input, is_prompt_processing) = match &pipeline.phase {
            Phase::PromptProcessing { pos, current_input } => (*pos, current_input.clone(), true),
            Phase::Verifying { pos, current_input, .. } => (*pos, current_input.clone(), false),
            _ => return,
        };
        let shard = pipeline.shards[pos].clone();
        let model_id = pipeline.manifest.model_id.clone();
        let total_layers = pipeline.manifest.num_layers;
        let backbone_pom_root = pipeline.manifest.backbone_pom_root.clone();
        let gguf_url = pipeline.source.gguf_url.clone();
        let tokenizer_url = pipeline.source.tokenizer_url.clone();
        let tokenizer_sha256 = pipeline.source.tokenizer_sha256.clone();
        drop(pipeline);

        let is_entry = pos == 0;
        let wallet = {
            let mut last = self.last_shard_wallet.lock();
            let Some(wallet) = self.registry.pick_next(&last) else {
                tracing::debug!(request_id = %request_id, "SpeculativeEngine: no miners connected right now, will retry next tick");
                return;
            };
            *last = Some(wallet.clone());
            wallet
        };
        let Some(tx) = self.registry.get(&wallet) else { return };

        let assign = OpoiShardAssign {
            request_id: request_id.to_string(),
            model_id,
            shard_index: shard.shard_index,
            layer_start: shard.layer_start.unwrap_or(0),
            layer_end: shard.layer_end.unwrap_or(0),
            total_layers,
            // Deliberately NEVER let the miner's own token_index/max_tokens
            // "is this the last token" check (`shard_pool_client.rs`'s
            // `compute()`) decide to finalize on our behalf — this engine
            // finalizes EXCLUSIVELY via an explicit `OpoiKvRollback{finalize:
            // true}` at the end of the round that reaches `max_tokens` (see
            // `handle_kv_rollback_ack_inner`). `token_index=0,
            // max_tokens=u32::MAX` guarantees `token_index+1>=max_tokens` is
            // never true on the miner side.
            token_index: 0,
            max_tokens: u32::MAX,
            input: current_input,
            model_gguf_url: gguf_url,
            model_gguf_sha256: backbone_pom_root,
            model_tokenizer_url: (is_entry && is_prompt_processing).then_some(tokenizer_url),
            model_tokenizer_sha256: (is_entry && is_prompt_processing).then_some(tokenizer_sha256),
        };
        self.shard_assignments.insert(request_id.to_string(), ShardStepAssignment { wallet: wallet.clone(), shard_index: shard.shard_index });
        let _ = tx.send(build_shard_assign_line(&assign));
        tracing::info!(request_id = %request_id, shard_index = shard.shard_index, wallet = %wallet, prompt_processing = is_prompt_processing, "speculative: dispatched shard step");
    }

    pub async fn handle_shard_result(&self, wallet: &str, result: OpoiShardResult) -> Result<(), crate::error::AppError> {
        let request_id = result.request_id.clone();
        {
            let Some(assignment) = self.shard_assignments.get(&request_id) else { return Err(crate::error::AppError::UnknownRequest) };
            if assignment.wallet != wallet || assignment.shard_index != result.shard_index {
                return Err(crate::error::AppError::NotAssignedToCaller);
            }
        }
        self.shard_assignments.remove(&request_id);

        let is_last_shard = {
            let Some(pipeline) = self.pipelines.get(&request_id) else { return Err(crate::error::AppError::UnknownRequest) };
            let pos = match &pipeline.phase {
                Phase::PromptProcessing { pos, .. } | Phase::Verifying { pos, .. } => *pos,
                _ => return Ok(()), // stale result for a phase we've already moved past
            };
            pos + 1 == pipeline.shards.len()
        };

        if !is_last_shard {
            if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
                let new_input = ShardInputWire::Tensor { shape: result.output.shape.clone(), data_hex: result.output.data_hex.clone() };
                match &mut pipeline.phase {
                    Phase::PromptProcessing { pos, current_input } => {
                        *pos += 1;
                        *current_input = new_input;
                    }
                    Phase::Verifying { pos, current_input, .. } => {
                        *pos += 1;
                        *current_input = new_input;
                    }
                    _ => {}
                }
            }
            self.dispatch_shard_step(&request_id).await;
            return Ok(());
        }

        // Last shard of this pass — behavior depends on which phase we're finishing.
        let was_prompt_processing = self.pipelines.get(&request_id).map(|p| matches!(p.phase, Phase::PromptProcessing { .. })).unwrap_or(false);

        if was_prompt_processing {
            let Some(t0) = result.next_token_id else {
                tracing::error!(request_id = %request_id, "speculative: prompt-processing result carried no next_token_id — dropping pipeline");
                self.pipelines.remove(&request_id);
                return Ok(());
            };
            // The bridge itself needs the REAL prompt length (in tokens) to
            // seed `base` — it never tokenizes locally elsewhere, but this
            // is the one place it must (see module doc). Resolved via the
            // same tokenizer URL/hash `ModelSource` already carries.
            let (tokenizer_url, tokenizer_sha256, prompt_hex) = {
                let Some(pipeline) = self.pipelines.get(&request_id) else { return Ok(()) };
                let prompt_hex = match &pipeline.phase {
                    Phase::PromptProcessing { current_input: ShardInputWire::Prompt { prompt_hex }, .. } => prompt_hex.clone(),
                    _ => String::new(),
                };
                (pipeline.source.tokenizer_url.clone(), pipeline.source.tokenizer_sha256.clone(), prompt_hex)
            };
            let base = match count_prompt_tokens(&self.tokenizer_cache_dir, &tokenizer_url, &tokenizer_sha256, &prompt_hex).await {
                Ok(n) => n,
                Err(e) => {
                    tracing::error!(request_id = %request_id, error = %e, "speculative: could not determine prompt token count — dropping pipeline");
                    self.pipelines.remove(&request_id);
                    return Ok(());
                }
            };

            if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
                pipeline.base = base;
                pipeline.expected0 = t0;
                pipeline.phase = Phase::AwaitingDraft;
            }
            self.dispatch_draft(&request_id).await;
            return Ok(());
        }

        // Finishing a VERIFY pass.
        let Some(target_logits) = result.next_token_ids else {
            tracing::error!(request_id = %request_id, "speculative: verify result carried no next_token_ids — dropping pipeline");
            self.pipelines.remove(&request_id);
            return Ok(());
        };

        let (carry, expected0, draft_token_ids, base, k, max_tokens) = {
            let Some(pipeline) = self.pipelines.get(&request_id) else { return Ok(()) };
            let draft_token_ids = match &pipeline.phase {
                Phase::Verifying { draft_token_ids, .. } => draft_token_ids.clone(),
                _ => return Ok(()),
            };
            (pipeline.carry, pipeline.expected0, draft_token_ids, pipeline.base, pipeline.k, pipeline.max_tokens)
        };

        let outcome = greedy_verify_round(carry, expected0, &draft_token_ids, &target_logits);
        let k_this_round = draft_token_ids.len();
        let keep_len = kv_keep_len(base, carry, outcome.accepted_count);
        let is_finalizing_round = (base + outcome.accepted_count + 1) as u32 >= max_tokens;
        let round = self.pipelines.get(&request_id).map(|p| p.round).unwrap_or(0);

        tracing::info!(
            request_id = %request_id, round,
            k_requested = k_this_round, accepted = outcome.accepted_count, full_accept = outcome.full_accept,
            settle = outcome.settle_token, finalizing = is_finalizing_round,
            "speculative: verify batch resolved"
        );

        let _ = k; // k for the NEXT round is derived once the rollback round fully completes — see handle_kv_rollback_ack.
        self.begin_rollback(&request_id, outcome, keep_len, k_this_round, is_finalizing_round).await;
        Ok(())
    }

    // ── draft dispatch ───────────────────────────────────────────────────

    async fn dispatch_draft(&self, request_id: &str) {
        let Some(mut pipeline) = self.pipelines.get_mut(request_id) else { return };
        if !matches!(pipeline.phase, Phase::AwaitingDraft) {
            return;
        }

        let wallet = match &pipeline.draft_wallet {
            Some(w) if self.registry.get(w).is_some() => w.clone(),
            _ => {
                let Some(w) = self.registry.pick_draft_capable(&None) else {
                    tracing::debug!(request_id = %request_id, "speculative: no draft-capable miner connected right now, will retry next tick");
                    return;
                };
                pipeline.draft_wallet = Some(w.clone());
                w
            }
        };
        let Some(tx) = self.registry.get(&wallet) else { return };

        let carry = pipeline.carry;
        let k = pipeline.k;
        let k_this_round = if carry.is_some() { (k as usize).saturating_sub(1).max(1) as u32 } else { k };
        let input = match carry {
            Some(c) => DraftInputWire::NextTokenId { token_id: c },
            None => DraftInputWire::Prompt { prompt_hex: pipeline.prompt_hex.clone() },
        };

        let draft = OpoiDraftGenerate {
            request_id: request_id.to_string(),
            model_id: pipeline.draft_source.draft_model_id.clone(),
            total_layers: pipeline.draft_source.draft_total_layers,
            k: k_this_round,
            input,
            model_gguf_url: pipeline.draft_source.draft_gguf_url.clone(),
            model_gguf_sha256: pipeline.draft_source.draft_gguf_sha256.clone(),
            model_tokenizer_url: (pipeline.round == 0).then(|| pipeline.draft_source.draft_tokenizer_url.clone()),
            model_tokenizer_sha256: (pipeline.round == 0).then(|| pipeline.draft_source.draft_tokenizer_sha256.clone()),
        };
        drop(pipeline);

        self.draft_assignments.insert(request_id.to_string(), wallet.clone());
        let _ = tx.send(build_draft_generate_line(&draft));
        tracing::info!(request_id = %request_id, wallet = %wallet, k = k_this_round, "speculative: dispatched draft-generate");
    }

    pub async fn handle_draft_result(&self, wallet: &str, result: OpoiDraftResult) {
        let request_id = result.request_id.clone();
        match self.draft_assignments.get(&request_id) {
            Some(a) if a.value() == wallet => {}
            _ => {
                tracing::warn!(request_id = %request_id, wallet = %wallet, "speculative: unexpected opoi.draft_result (not the current assignment); dropping");
                return;
            }
        }
        self.draft_assignments.remove(&request_id);

        if result.draft_token_ids.is_empty() {
            tracing::warn!(request_id = %request_id, "speculative: draft-generate returned no tokens — retrying next tick");
            return; // stays in AwaitingDraft; retry_stalled_tick will redispatch
        }

        let current_input = {
            let Some(pipeline) = self.pipelines.get(&request_id) else { return };
            let mut batch: Vec<u32> = Vec::with_capacity(result.draft_token_ids.len() + 1);
            if let Some(c) = pipeline.carry {
                batch.push(c);
            }
            batch.extend_from_slice(&result.draft_token_ids);
            ShardInputWire::TokenIds { token_ids: batch }
        };

        if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
            pipeline.round += 1;
            pipeline.phase = Phase::Verifying { pos: 0, current_input, draft_token_ids: result.draft_token_ids };
        }
        self.dispatch_shard_step(&request_id).await;
    }

    // ── rollback ──────────────────────────────────────────────────────────

    async fn begin_rollback(&self, request_id: &str, outcome: RoundOutcome, keep_len: usize, k_this_round: usize, is_finalizing_round: bool) {
        let Some(pipeline) = self.pipelines.get(request_id) else { return };
        let mut pending = HashSet::new();
        for shard in &pipeline.shards {
            let rollback = OpoiKvRollback {
                request_id: request_id.to_string(),
                shard_index: shard.shard_index,
                keep_until_pos: keep_len as u32,
                finalize: is_finalizing_round,
            };
            // Same miner that JUST returned this round's result for that
            // shard — `shard_assignments` was already cleared for this
            // request (verify pass done), so we rely on round-robin
            // dispatch history instead: in practice the SAME wallet handled
            // every shard of this round's verify pass (dispatch_shard_step
            // picks fresh each time, but a small pool converges quickly).
            // Simplification (documented): route every shard's rollback
            // through `pick_next`'s current wallet if the exact one isn't
            // trivially recoverable — acceptable because `OpoiKvRollback`'s
            // effect (truncate/finalize) only depends on shard_index, not
            // which miner asked, PROVIDED it's the miner actually holding
            // that shard's session (best-effort here; a wrong target simply
            // acks `ok:false`, logged, not fatal to the rest of the round).
            if let Some(wallet) = self.registry.pick_next(&self.last_shard_wallet.lock()) {
                if let Some(tx) = self.registry.get(&wallet) {
                    let _ = tx.send(build_kv_rollback_line(&rollback));
                }
            }
            pending.insert(shard.shard_index);
        }
        if let Some(draft_wallet) = &pipeline.draft_wallet {
            if let Some(tx) = self.registry.get(draft_wallet) {
                let rollback = OpoiKvRollback { request_id: request_id.to_string(), shard_index: DRAFT_SESSION_SHARD_INDEX, keep_until_pos: keep_len as u32, finalize: false };
                let _ = tx.send(build_kv_rollback_line(&rollback));
            }
        }
        pending.insert(DRAFT_SESSION_SHARD_INDEX);
        drop(pipeline);

        if let Some(mut pipeline) = self.pipelines.get_mut(request_id) {
            pipeline.phase = Phase::AwaitingRollback(RollbackState {
                pending,
                accepted_count: outcome.accepted_count,
                k_this_round,
                settle_token: outcome.settle_token,
                keep_len,
                is_finalizing_round,
            });
        }
    }

    pub async fn handle_kv_rollback_ack(&self, _wallet: &str, ack: OpoiKvRollbackAck) {
        let request_id = ack.request_id.clone();

        let round_complete = {
            let Some(mut pipeline) = self.pipelines.get_mut(&request_id) else { return };
            let Phase::AwaitingRollback(state) = &mut pipeline.phase else { return };
            if !ack.ok {
                tracing::warn!(request_id = %request_id, shard_index = ack.shard_index, "speculative: kv_rollback_ack reported failure — pipeline may be stuck (no retry for this yet)");
            }
            if ack.shard_index != DRAFT_SESSION_SHARD_INDEX && state.is_finalizing_round && ack.finalized_hash.is_some() {
                // Submit on-chain immediately for this shard — every real
                // shard finalizes independently, mirroring
                // shard_engine.rs's own per-shard on-chain submission.
                let hash = ack.finalized_hash.clone().unwrap();
                let shard_index = ack.shard_index;
                let csd = self.csd.clone();
                let stake_pool = self.stake_pool.clone();
                let req_id = request_id.clone();
                tokio::spawn(async move {
                    let submit = stake_pool
                        .try_each(|addr| {
                            let csd = csd.clone();
                            let req_id = req_id.clone();
                            let hash = hash.clone();
                            async move { csd.submit_shard_result(&req_id, shard_index, &addr, &hash, 0).await }
                        })
                        .await;
                    if let Err(e) = submit {
                        tracing::warn!(request_id = %req_id, shard_index, error = %e, "speculative: no pool address eligible to submit this shard result");
                    }
                });
            }
            state.pending.remove(&ack.shard_index);
            state.pending.is_empty()
        };

        if !round_complete {
            return;
        }

        let (is_finalizing_round, keep_len, settle_token, accepted_count, k_this_round) = {
            let Some(pipeline) = self.pipelines.get(&request_id) else { return };
            let Phase::AwaitingRollback(state) = &pipeline.phase else { return };
            (state.is_finalizing_round, state.keep_len, state.settle_token, state.accepted_count, state.k_this_round)
        };

        if is_finalizing_round {
            tracing::info!(request_id = %request_id, "speculative: pipeline finished (max_tokens reached)");
            self.pipelines.remove(&request_id);
            return;
        }

        if let Some(mut pipeline) = self.pipelines.get_mut(&request_id) {
            pipeline.base = keep_len;
            pipeline.carry = Some(settle_token);
            pipeline.expected0 = settle_token;
            pipeline.k = next_k(pipeline.k, accepted_count, k_this_round);
            pipeline.phase = Phase::AwaitingDraft;
        }
        self.dispatch_draft(&request_id).await;
    }

    pub async fn on_disconnect(&self, wallet: &str) {
        self.shard_assignments.retain(|_, a| a.wallet != wallet);
        self.draft_assignments.retain(|_, w| w != wallet);
        // A pipeline whose PINNED draft wallet just disconnected can't keep
        // speculating (see module doc) — drop it rather than silently
        // stalling forever; ShardEngine's own plain path will pick this
        // request back up fresh from PENDING on csd's side IF the request
        // is still outstanding there (out of this session's scope to wire
        // that hand-back automatically).
        let lost: Vec<String> = self
            .pipelines
            .iter()
            .filter(|e| e.draft_wallet.as_deref() == Some(wallet))
            .map(|e| e.key().clone())
            .collect();
        for request_id in lost {
            tracing::warn!(request_id = %request_id, wallet = %wallet, "speculative: pinned draft-capable miner disconnected — dropping pipeline (see module doc's known limitation)");
            self.pipelines.remove(&request_id);
        }
    }
}

/// Downloads (once, cached by sha256) and tokenizes `prompt_hex` to count
/// its REAL token length — the one place this bridge needs to tokenize
/// anything locally (see module doc: `base` must be seeded with the true
/// prompt length, and nothing else in `OpoiShardResult` exposes it once the
/// entry shard's output collapses to a single-position projection).
async fn count_prompt_tokens(cache_dir: &std::path::Path, url: &str, sha256_hex: &str, prompt_hex: &str) -> anyhow::Result<usize> {
    let prompt = String::from_utf8(hex::decode(prompt_hex)?)?;
    let path = ensure_tokenizer_cached(cache_dir, url, sha256_hex).await?;
    let path_owned = path.clone();
    let prompt_owned = prompt.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
        let tokenizer = tokenizers::Tokenizer::from_file(&path_owned).map_err(|e| anyhow::anyhow!("{e}"))?;
        let encoding = tokenizer.encode(prompt_owned.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(encoding.get_ids().len())
    })
    .await?
}

#[async_trait::async_trait]
impl crate::opoi::handler::SpeculativeHandler for SpeculativeEngine {
    async fn handle_draft_result(&self, wallet: &str, result: OpoiDraftResult) {
        self.handle_draft_result(wallet, result).await
    }

    async fn handle_kv_rollback_ack(&self, wallet: &str, ack: OpoiKvRollbackAck) {
        self.handle_kv_rollback_ack(wallet, ack).await
    }

    async fn on_disconnect(&self, wallet: &str) {
        self.on_disconnect(wallet).await
    }
}

async fn ensure_tokenizer_cached(cache_dir: &std::path::Path, url: &str, sha256_hex: &str) -> anyhow::Result<PathBuf> {
    tokio::fs::create_dir_all(cache_dir).await.ok();
    let path = cache_dir.join(format!("{sha256_hex}.tokenizer.json"));
    if path.exists() {
        return Ok(path);
    }
    let bytes = reqwest::get(url).await?.bytes().await?;
    let actual = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
    if actual != sha256_hex {
        anyhow::bail!("tokenizer at {url} hash mismatch: expected {sha256_hex}, got {actual}");
    }
    tokio::fs::write(&path, &bytes).await?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors cs-miner's src/opoi/speculative_round.rs test cases
    // byte-for-byte — this port must behave identically since the two
    // engines run the SAME protocol against each other in production.

    #[test]
    fn round0_full_accept() {
        let out = greedy_verify_round(None, 11, &[11, 12, 13], &[12, 13, 14]);
        assert_eq!(out, RoundOutcome { accepted_count: 3, settle_token: 14, full_accept: true });
    }

    #[test]
    fn round0_reject_at_first_position() {
        let out = greedy_verify_round(None, 999, &[10, 11], &[5, 6]);
        assert_eq!(out, RoundOutcome { accepted_count: 0, settle_token: 999, full_accept: false });
    }

    #[test]
    fn round_ge1_carry_trivially_accepted_then_full_accept() {
        let out = greedy_verify_round(Some(7), 0, &[100, 101, 102], &[100, 101, 102, 999]);
        assert_eq!(out, RoundOutcome { accepted_count: 3, settle_token: 999, full_accept: true });
    }

    #[test]
    fn kv_keep_len_round_ge1_includes_carry() {
        assert_eq!(kv_keep_len(50, Some(42), 3), 54);
    }

    #[test]
    fn kv_keep_len_round0_matches_doc_formula() {
        assert_eq!(kv_keep_len(50, None, 3), 53);
    }

    #[test]
    fn next_k_matches_cs_miner() {
        assert_eq!(next_k(4, 4, 4), 5);
        assert_eq!(next_k(4, 1, 4), 3);
        assert_eq!(next_k(8, 8, 8), 8);
        assert_eq!(next_k(2, 0, 4), 2);
    }

    #[test]
    fn draft_capable_registry_round_robin_and_cleanup() {
        let registry = MinerRegistry::new();
        let (tx_a, _rx_a) = tokio::sync::mpsc::unbounded_channel();
        let (tx_b, _rx_b) = tokio::sync::mpsc::unbounded_channel();
        registry.register("wallet_a".to_string(), tx_a);
        registry.register("wallet_b".to_string(), tx_b);

        // Neither marked draft-capable yet.
        assert_eq!(registry.pick_draft_capable(&None), None);

        registry.mark_draft_capable("wallet_a");
        assert_eq!(registry.pick_draft_capable(&None), Some("wallet_a".to_string()));

        registry.mark_draft_capable("wallet_b");
        // Round-robins past wallet_a to wallet_b.
        assert_eq!(registry.pick_draft_capable(&Some("wallet_a".to_string())), Some("wallet_b".to_string()));

        // Disconnecting a draft-capable wallet clears its eligibility.
        registry.unregister("wallet_b");
        assert_eq!(registry.pick_draft_capable(&Some("wallet_a".to_string())), Some("wallet_a".to_string()));
    }
}
