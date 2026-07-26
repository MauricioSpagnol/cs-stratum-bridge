//! D2 follow-up (2026-07-26): real per-expert VRAM sizing for
//! `ExpertDispatcher::dispatch_and_join`'s `min_vram_mb_for` callback — see
//! `shard_engine.rs::handle_expert_group_result`, the only call site, and
//! `miner_registry.rs::pick_expert_hosts_top_n`'s doc comment for the TODO
//! this resolves (`min_vram_mb` was hardcoded to `0`, i.e. no floor at all,
//! since "GGUF/manifest expert size metadata isn't threaded through to this
//! layer yet").
//!
//! ## Why this reads the GGUF itself, not `getmodelmanifest`
//!
//! `ModelManifest` (`rpc/types.rs`) doesn't carry a per-expert size, and the
//! one manifest field that comes closest — `total_size_bytes` (on-chain,
//! `opoi_model_manifest.h`, proposer-declared, optional, default 0 = "not
//! declared") — is the WHOLE model's size (attention + embeddings + lm_head
//! + every layer's experts combined), not just one layer's FFN-expert slice,
//! and isn't guaranteed to even be declared. Deriving a per-expert estimate
//! from it would mean guessing what fraction of the total is "this layer's
//! experts" — a genuinely coarse heuristic, and a dangerous one to get
//! wrong in the UNDER-estimate direction (an accepted host that's actually
//! too small OOMs mid-dispatch). The GGUF file itself, by contrast, states
//! each `blk.{layer}.ffn_{gate,up,down}_exps.weight` tensor's exact shape
//! and quantization — the SAME bytes a real miner loads via cs-miner's
//! `shard_model_moe.rs::MoeModelWeights::from_gguf` — so reading it is exact,
//! not a guess.
//!
//! ## Why this doesn't depend on `candle-core` (cs-miner's own GGUF parser)
//!
//! cs-miner already has exactly the right pure functions
//! (`shard_model_moe.rs::moe_expert_vram_bytes`/`_mb`), but they take a
//! `candle_core::quantized::gguf_file::Content`, and `candle-core` pulls in
//! ~125 transitive crates (measured via `cargo tree -p candle-core` in
//! cs-miner: `half`, `gemm`, `rayon`, `safetensors`, `memmap2`, optional
//! `cudarc`, ...) — a heavy dependency tree for a process (this bridge) that
//! never runs a tensor op, just needs a handful of tensor *shapes* out of a
//! file header. Since a GGUF's header (magic/version/counts, metadata KV
//! pairs, tensor-info table) is a small, stable, fully-documented binary
//! format — and reading it needs no candle types, just the same `(name,
//! dims, ggml_type)` triples `gguf_file::Content::read` itself extracts —
//! this module hand-parses just that header instead. Validated against a
//! REAL MoE GGUF locally (`TINYMIXTRAL.gguf`, 430MB total, 12 layers/4
//! experts): its header — 43 metadata KVs including a 32003-entry tokenizer
//! vocab array — is only ~727KB, confirming `HEADER_PREFIX_CAP_BYTES` below
//! (16MiB) has generous headroom even for much larger-vocab models, while
//! never reading anywhere near the full multi-GB tensor-data section.
//!
//! ## Fetch strategy
//!
//! `fetch_gguf_header_prefix` sends a `Range: bytes=0-N` request (a courtesy
//! to servers that honor it — cheap, precise) but doesn't rely on the
//! server actually honoring it: it reads the response via `Response::chunk`
//! (available without reqwest's `stream` cargo feature, so no Cargo.toml
//! change is needed either) and simply stops — dropping the response,
//! closing the connection — the moment it's accumulated `cap_bytes`,
//! regardless of whether the server sent `206 Partial Content` or ignored
//! the header and started streaming a full multi-GB `200 OK` body. Either
//! way, at most `cap_bytes` is ever actually read off the wire.
//!
//! ## Caching
//!
//! One header fetch yields every `blk.N`'s tensor info at once, so
//! `ExpertVramEstimator` fetches/parses a given `(model_id,
//! gguf_sha256)` — content-addressed, safe to cache forever within this
//! process's lifetime — at most once, then serves every subsequent
//! `min_vram_mb_for_range` call (one per token per MoE layer-group
//! dispatch — a hot path) out of an in-memory per-layer byte-size map.
//!
//! ## Layer-range granularity
//!
//! A MoE manifest's dense-boundary `(layer_start, layer_end)` range CAN
//! legally span more than one real GGUF `blk.N` (cloudservice's
//! `SplitLayerRanges` is a plain even split, not required to equal
//! `num_layers` many ranges — see `shard_engine.rs`'s module doc and
//! cs-miner's D2 session report for the full derivation), so a single
//! `handle_expert_group_result` call's range may span several layers at
//! once, not just one — even though (2026-07-26 multi-layer follow-up
//! session) that call currently only externally dispatches the range's
//! LAST layer's selection. `min_vram_mb_for_range` handles any range width
//! uniformly regardless: it takes the MAX per-expert byte size over every
//! `blk.N` in `[layer_start, layer_end)` — a safe (conservative, never an
//! under-estimate) floor either way, and exactly right in the common case
//! that every layer's experts are the same size (true for every standard
//! llama.cpp MoE GGUF layout: `ffn_*_exps.weight` tensors don't vary
//! shape/dtype across layers).

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;

/// Generous cap on how much of a GGUF's head this module will ever read —
/// see this module's doc comment for why a real 430MB MoE GGUF's actual
/// header was only ~727KB. If a real model's metadata (e.g. an enormous
/// tokenizer vocab) ever pushes the true header past this, `fetch_and_parse`
/// fails cleanly (see `ParseErr::Incomplete`'s mapping) and
/// `min_vram_mb_for_range` falls back to `None` (no floor) — never a wrong
/// answer, just a missing one.
const HEADER_PREFIX_CAP_BYTES: usize = 16 * 1024 * 1024;

/// Defensive sanity cap on `tensor_count`/`metadata_kv_count` — both are
/// attacker/misconfiguration-controllable `u64`s read directly off the wire
/// before anything is validated. A real GGUF never has anywhere near this
/// many of either; this just bounds how much CPU a garbled response can
/// burn before `parse_gguf_tensor_infos` gives up (each iteration still
/// consumes real bytes from the capped buffer above, so the real ceiling is
/// already low — this is belt-and-suspenders, not the primary defense).
const MAX_METADATA_OR_TENSOR_COUNT: u64 = 1_000_000;

/// One tensor's shape + GGML quantization type, exactly what
/// `expert_vram_bytes_for_layer` needs — the header's `offset` field is read
/// (to stay positioned correctly for the next tensor info) but not kept:
/// this module never reads tensor DATA, only shapes, so a tensor's byte
/// offset into the data section is irrelevant here.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GgufTensorInfo {
    dims: Vec<u64>,
    ggml_type: u32,
}

/// Distinguishes "the buffer just isn't big enough yet" (caller could, in
/// principle, fetch more) from "this isn't a valid/recognized GGUF at all"
/// (more bytes would never help) — `fetch_and_parse` folds both into a
/// single `String` error for its caller (this module deliberately doesn't
/// retry with a bigger buffer — see `HEADER_PREFIX_CAP_BYTES`'s doc comment
/// for why one generous fixed-size prefix is preferred over incremental
/// grow-and-retry complexity), but keeping them distinct here makes the
/// unit tests that exercise "truncated input" vs "malformed input"
/// unambiguous about which they're checking.
#[derive(Debug, PartialEq, Eq)]
enum ParseErr {
    Incomplete,
    Invalid(String),
}

fn read_exact_or_incomplete(cur: &mut Cursor<&[u8]>, buf: &mut [u8]) -> Result<(), ParseErr> {
    cur.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            ParseErr::Incomplete
        } else {
            ParseErr::Invalid(e.to_string())
        }
    })
}

fn read_u32(cur: &mut Cursor<&[u8]>) -> Result<u32, ParseErr> {
    let mut buf = [0u8; 4];
    read_exact_or_incomplete(cur, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(cur: &mut Cursor<&[u8]>) -> Result<u64, ParseErr> {
    let mut buf = [0u8; 8];
    read_exact_or_incomplete(cur, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// GGUF's `gguf_string_t`: a `u64` byte length followed by (NOT
/// necessarily NUL-terminated) UTF-8 bytes. `from_utf8_lossy` rather than a
/// hard error on invalid UTF-8 — a key/value's text isn't itself
/// load-bearing for this module (only tensor NAMES are matched against, and
/// those are always plain ASCII in every real GGUF), so tolerating garbage
/// here rather than failing the whole header parse over an unrelated
/// metadata string is the more robust choice.
fn read_gguf_string(cur: &mut Cursor<&[u8]>) -> Result<String, ParseErr> {
    let len = read_u64(cur)?;
    if len > HEADER_PREFIX_CAP_BYTES as u64 {
        return Err(ParseErr::Invalid(format!("gguf string length {len} exceeds header prefix cap")));
    }
    let mut buf = vec![0u8; len as usize];
    read_exact_or_incomplete(cur, &mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Skips one metadata value of `vtype` (the GGUF metadata value-type enum:
/// 0=u8,1=i8,2=u16,3=i16,4=u32,5=i32,6=f32,7=bool,8=string,9=array,
/// 10=u64,11=i64,12=f64) without interpreting it — this module has no use
/// for any metadata VALUE, only for correctly walking past every one of
/// them to reach the tensor-info table that follows. Recurses for
/// `ARRAY` (element type + count + that many elements, themselves
/// recursively skipped — handles an array of strings, or in principle an
/// array of arrays, the same way).
fn skip_metadata_value(cur: &mut Cursor<&[u8]>, vtype: u32) -> Result<(), ParseErr> {
    match vtype {
        0 | 1 | 7 => read_exact_or_incomplete(cur, &mut [0u8; 1]),
        2 | 3 => read_exact_or_incomplete(cur, &mut [0u8; 2]),
        4 | 5 | 6 => read_exact_or_incomplete(cur, &mut [0u8; 4]),
        10 | 11 | 12 => read_exact_or_incomplete(cur, &mut [0u8; 8]),
        8 => read_gguf_string(cur).map(|_| ()),
        9 => {
            let elem_type = read_u32(cur)?;
            let len = read_u64(cur)?;
            if len > MAX_METADATA_OR_TENSOR_COUNT {
                return Err(ParseErr::Invalid(format!("metadata array length {len} exceeds sanity cap")));
            }
            for _ in 0..len {
                skip_metadata_value(cur, elem_type)?;
            }
            Ok(())
        }
        other => Err(ParseErr::Invalid(format!("unknown metadata value type {other}"))),
    }
}

/// Parses a GGUF header prefix into every tensor's `(name -> shape,
/// ggml_type)` — mirrors `candle_core::quantized::gguf_file::Content::read`'s
/// header-parsing half exactly (magic, version, counts, metadata KVs,
/// tensor-info table), but stops there: never touches the tensor DATA
/// section (which starts right after this, alignment-padded) since nothing
/// here needs it. `bytes` need only cover the header, not the whole file —
/// see this module's doc comment for why `HEADER_PREFIX_CAP_BYTES` is
/// generous enough for that in practice.
fn parse_gguf_tensor_infos(bytes: &[u8]) -> Result<HashMap<String, GgufTensorInfo>, ParseErr> {
    let mut cur = Cursor::new(bytes);

    let mut magic = [0u8; 4];
    read_exact_or_incomplete(&mut cur, &mut magic)?;
    if &magic != b"GGUF" {
        return Err(ParseErr::Invalid(format!("bad GGUF magic bytes {magic:?}")));
    }
    let _version = read_u32(&mut cur)?;
    let tensor_count = read_u64(&mut cur)?;
    let kv_count = read_u64(&mut cur)?;
    if tensor_count > MAX_METADATA_OR_TENSOR_COUNT || kv_count > MAX_METADATA_OR_TENSOR_COUNT {
        return Err(ParseErr::Invalid(format!("implausible tensor_count={tensor_count}/kv_count={kv_count}")));
    }

    for _ in 0..kv_count {
        let _key = read_gguf_string(&mut cur)?;
        let vtype = read_u32(&mut cur)?;
        skip_metadata_value(&mut cur, vtype)?;
    }

    let mut infos = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut cur)?;
        let n_dims = read_u32(&mut cur)?;
        if n_dims as u64 > MAX_METADATA_OR_TENSOR_COUNT {
            return Err(ParseErr::Invalid(format!("implausible n_dims={n_dims} for tensor {name}")));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut cur)?);
        }
        let ggml_type = read_u32(&mut cur)?;
        let _offset = read_u64(&mut cur)?; // data-section offset — irrelevant here, see GgufTensorInfo's doc.
        infos.insert(name, GgufTensorInfo { dims, ggml_type });
    }

    Ok(infos)
}

/// `(block_size, type_size_bytes)` for every GGML quantization type
/// `candle-core` 0.11.0 (the crate real miners load GGUFs through —
/// `cs-miner/src/opoi/shard_model_moe.rs`) can actually load — mirrors
/// `GgmlDType::{from_u32,block_size,type_size}` in candle-core's
/// `src/quantized/mod.rs` verbatim (numeric ids and byte sizes cross-checked
/// against that file's own `Block*` struct definitions and their
/// `const _: () = assert!(size_of::<Block*>() == N)` compile-time checks).
/// Deliberately the SAME set candle-core supports, not a superset: a GGUF
/// tensor using a dtype outside this list couldn't be loaded by a real
/// miner either, so erroring here (rather than guessing) is the right
/// failure mode.
fn ggml_block_and_type_size(ggml_type: u32) -> Option<(u64, u64)> {
    match ggml_type {
        0 => Some((1, 4)),      // F32
        1 => Some((1, 2)),      // F16
        2 => Some((32, 18)),    // Q4_0
        3 => Some((32, 20)),    // Q4_1
        6 => Some((32, 22)),    // Q5_0
        7 => Some((32, 24)),    // Q5_1
        8 => Some((32, 34)),    // Q8_0
        9 => Some((32, 36)),    // Q8_1
        10 => Some((256, 84)),  // Q2_K
        11 => Some((256, 110)), // Q3_K
        12 => Some((256, 144)), // Q4_K
        13 => Some((256, 176)), // Q5_K
        14 => Some((256, 210)), // Q6_K
        15 => Some((256, 292)), // Q8_K
        30 => Some((1, 2)),     // BF16
        _ => None,
    }
}

/// One layer's per-expert VRAM footprint in bytes — sums the three stacked
/// `(n_expert, out_dim, in_dim)` FFN-expert tensors (`ffn_gate_exps`,
/// `ffn_up_exps`, `ffn_down_exps`), each divided by `n_expert`, exactly
/// mirroring cs-miner's `shard_model_moe.rs::moe_expert_vram_bytes` (same
/// three tensor-name suffixes, same divisibility checks against `n_expert`
/// and each dtype's block size) — this is a from-scratch reimplementation
/// (this crate can't depend on cs-miner), not a copy-paste, but computes the
/// identical quantity from the identical header fields.
fn expert_vram_bytes_for_layer(tensor_infos: &HashMap<String, GgufTensorInfo>, layer_idx: u32, n_expert: u64) -> Result<u64, String> {
    if n_expert == 0 {
        return Err("expert_vram_bytes_for_layer: n_expert must be > 0".to_string());
    }
    let mut total_bytes = 0u64;
    for suffix in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        let name = format!("blk.{layer_idx}.{suffix}.weight");
        let info = tensor_infos.get(&name).ok_or_else(|| format!("cannot find tensor info for {name}"))?;
        let total_elems: u64 = info.dims.iter().product();
        if total_elems % n_expert != 0 {
            return Err(format!("{name}'s elem_count {total_elems} is not divisible by n_expert {n_expert}"));
        }
        let per_expert_elems = total_elems / n_expert;
        let (block_size, type_size) =
            ggml_block_and_type_size(info.ggml_type).ok_or_else(|| format!("{name}: unsupported/unknown ggml_type {}", info.ggml_type))?;
        if per_expert_elems % block_size != 0 {
            return Err(format!("{name}'s per-expert elem_count {per_expert_elems} is not divisible by block_size {block_size}"));
        }
        total_bytes += (per_expert_elems / block_size) * type_size;
    }
    Ok(total_bytes)
}

/// Streams (at most) `cap_bytes` off the front of `url` — see this module's
/// doc comment for why this is safe regardless of whether the server honors
/// the `Range` request this sends as a courtesy. Returns whatever was
/// actually available if the resource is smaller than `cap_bytes` (a tiny
/// test fixture, say) rather than erroring.
async fn fetch_gguf_header_prefix(client: &reqwest::Client, url: &str, cap_bytes: usize) -> Result<Vec<u8>, String> {
    let mut resp = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes=0-{}", cap_bytes.saturating_sub(1)))
        .send()
        .await
        .map_err(|e| format!("GET {url} failed: {e}"))?;

    if !(resp.status().is_success() || resp.status() == reqwest::StatusCode::PARTIAL_CONTENT) {
        return Err(format!("GET {url}: HTTP {}", resp.status()));
    }

    let mut buf = Vec::with_capacity(cap_bytes.min(1 << 20));
    while buf.len() < cap_bytes {
        match resp.chunk().await.map_err(|e| format!("reading {url}: {e}"))? {
            Some(chunk) => buf.extend_from_slice(&chunk),
            None => break, // server's whole response was smaller than cap_bytes.
        }
    }
    buf.truncate(cap_bytes.min(buf.len()));
    // `resp` drops here — closes the connection even if the server ignored
    // `Range` and was mid-stream on a much larger body.
    Ok(buf)
}

/// Fetches (or serves from cache) every `blk.N`'s per-expert VRAM byte size
/// for one GGUF, and answers `min_vram_mb_for_range`'s "what's the floor for
/// this dispatch group" question from it. See this module's doc comment for
/// the full design (fetch strategy, caching, layer-range granularity).
pub struct ExpertVramEstimator {
    client: reqwest::Client,
    /// Key: `"{model_id}:{gguf_sha256}"` (content-addressed — safe to keep
    /// forever within this process's lifetime, see doc comment). Value:
    /// `blk.N`'s index -> that layer's per-expert byte size, for every `N`
    /// this GGUF's header actually has FFN-expert tensors for.
    cache: DashMap<String, Arc<HashMap<u32, u64>>>,
}

impl Default for ExpertVramEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertVramEstimator {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .expect("failed to build reqwest client");
        Self { client, cache: DashMap::new() }
    }

    /// Returns a safe (conservative — see doc comment) per-expert VRAM
    /// floor in MB for a fan-out group spanning `[layer_start, layer_end)`,
    /// or `None` if it couldn't be determined (network/parse failure, or no
    /// FFN-expert tensors found for any layer in range) — callers should
    /// treat `None` the same as the pre-D2-follow-up behavior: no floor
    /// (`0`), never a hard failure of the whole dispatch over a VRAM
    /// estimate being unavailable.
    pub async fn min_vram_mb_for_range(&self, model_id: &str, gguf_url: &str, gguf_sha256: &str, layer_start: u32, layer_end: u32, n_expert: u64) -> Option<u64> {
        let cache_key = format!("{model_id}:{gguf_sha256}");

        let per_layer = match self.cache.get(&cache_key) {
            Some(cached) => cached.clone(),
            None => match self.fetch_and_parse(gguf_url, n_expert).await {
                Ok(map) => {
                    let arc = Arc::new(map);
                    self.cache.insert(cache_key, arc.clone());
                    arc
                }
                Err(e) => {
                    tracing::warn!(
                        model_id, gguf_url, error = %e,
                        "D2: failed to estimate per-expert VRAM from GGUF header; dispatching with no VRAM floor (0), same as before this estimator existed"
                    );
                    return None;
                }
            },
        };

        let max_bytes = (layer_start..layer_end).filter_map(|l| per_layer.get(&l).copied()).max()?;
        Some(max_bytes.div_ceil(1024 * 1024))
    }

    async fn fetch_and_parse(&self, gguf_url: &str, n_expert: u64) -> Result<HashMap<u32, u64>, String> {
        let bytes = fetch_gguf_header_prefix(&self.client, gguf_url, HEADER_PREFIX_CAP_BYTES).await?;
        let tensor_infos = parse_gguf_tensor_infos(&bytes).map_err(|e| match e {
            ParseErr::Incomplete => format!("GGUF header did not fit within the {HEADER_PREFIX_CAP_BYTES}-byte prefix this module reads"),
            ParseErr::Invalid(msg) => msg,
        })?;

        // Standard llama.cpp GGUF layout numbers layers contiguously from 0
        // — walk forward until a layer's FFN-expert tensors are no longer
        // found, which (for a valid MoE GGUF) means we've walked past
        // `block_count`.
        let mut per_layer = HashMap::new();
        let mut layer_idx = 0u32;
        loop {
            match expert_vram_bytes_for_layer(&tensor_infos, layer_idx, n_expert) {
                Ok(bytes) => {
                    per_layer.insert(layer_idx, bytes);
                    layer_idx += 1;
                }
                Err(e) if layer_idx == 0 => {
                    return Err(format!("no blk.0 FFN-expert tensors found in GGUF header (not a MoE checkpoint, or header prefix too small): {e}"));
                }
                Err(_) => break,
            }
        }
        Ok(per_layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal but wire-valid GGUF byte buffer with `n_layer`
    /// layers' worth of `blk.N.ffn_{gate,up,down}_exps.weight` tensors
    /// (dims `[hidden, ffn, n_expert]` for gate/up, `[ffn, hidden,
    /// n_expert]` for down — matching real llama.cpp MoE GGUFs) plus one
    /// harmless scalar metadata KV, so tests don't depend on any real
    /// downloaded model file. `kv_count=1` (not 0) deliberately, so the
    /// metadata-skipping path (not just the empty case) gets exercised.
    fn build_synthetic_gguf(n_layer: u32, hidden: u64, ffn: u64, n_expert: u64, exps_dtype: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        let tensor_count = n_layer as u64 * 3;
        out.extend_from_slice(&tensor_count.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes()); // kv_count

        // One scalar metadata KV: "general.alignment" = 32u32 (vtype 4).
        write_gguf_string(&mut out, "general.alignment");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&32u32.to_le_bytes());

        let mut offset = 0u64;
        for layer in 0..n_layer {
            for (suffix, dims) in [
                ("ffn_gate_exps", vec![hidden, ffn, n_expert]),
                ("ffn_up_exps", vec![hidden, ffn, n_expert]),
                ("ffn_down_exps", vec![ffn, hidden, n_expert]),
            ] {
                write_gguf_string(&mut out, &format!("blk.{layer}.{suffix}.weight"));
                out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in &dims {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&exps_dtype.to_le_bytes());
                out.extend_from_slice(&offset.to_le_bytes());
                offset += 1024; // dummy, unread by anything meaningful
            }
        }
        out
    }

    fn write_gguf_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn ggml_block_and_type_size_matches_candle_core_table() {
        // Cross-checked against candle-core 0.11.0's own `BlockQ*`
        // `size_of` compile-time assertions (see this module's doc
        // comment) — Q4_K/Q6_K are the two actually exercised by the
        // synthetic-GGUF tests below.
        assert_eq!(ggml_block_and_type_size(0), Some((1, 4))); // F32
        assert_eq!(ggml_block_and_type_size(2), Some((32, 18))); // Q4_0
        assert_eq!(ggml_block_and_type_size(12), Some((256, 144))); // Q4_K
        assert_eq!(ggml_block_and_type_size(14), Some((256, 210))); // Q6_K
        assert_eq!(ggml_block_and_type_size(30), Some((1, 2))); // BF16
        assert_eq!(ggml_block_and_type_size(9999), None);
    }

    #[test]
    fn parse_gguf_tensor_infos_finds_every_expert_tensor() {
        let bytes = build_synthetic_gguf(2, 256, 512, 4, 12);
        let infos = parse_gguf_tensor_infos(&bytes).expect("valid synthetic gguf parses");
        assert_eq!(infos.len(), 6); // 2 layers * 3 tensors each
        let gate0 = &infos["blk.0.ffn_gate_exps.weight"];
        assert_eq!(gate0.dims, vec![256, 512, 4]);
        assert_eq!(gate0.ggml_type, 12);
        let down1 = &infos["blk.1.ffn_down_exps.weight"];
        assert_eq!(down1.dims, vec![512, 256, 4]);
    }

    #[test]
    fn parse_gguf_tensor_infos_rejects_bad_magic() {
        let mut bytes = build_synthetic_gguf(1, 64, 128, 2, 12);
        bytes[0] = b'X';
        assert!(matches!(parse_gguf_tensor_infos(&bytes), Err(ParseErr::Invalid(_))));
    }

    #[test]
    fn parse_gguf_tensor_infos_reports_incomplete_on_truncation() {
        let bytes = build_synthetic_gguf(2, 256, 512, 4, 12);
        let truncated = &bytes[..bytes.len() / 2];
        assert_eq!(parse_gguf_tensor_infos(truncated), Err(ParseErr::Incomplete));
    }

    #[test]
    fn expert_vram_bytes_for_layer_matches_hand_computed_value() {
        // hidden=256, ffn=512, n_expert=4, gate/up=Q4_K(12), down=Q6_K(14).
        // Per-expert elem_count = 256*512*4/4 = 131072 for every tensor.
        // Q4_K: 131072/256 blocks * 144 bytes/block = 512*144 = 73728.
        // Q6_K: 131072/256 blocks * 210 bytes/block = 512*210 = 107520.
        // Total = 73728*2 + 107520 = 254976.
        let mut infos = HashMap::new();
        infos.insert("blk.0.ffn_gate_exps.weight".to_string(), GgufTensorInfo { dims: vec![256, 512, 4], ggml_type: 12 });
        infos.insert("blk.0.ffn_up_exps.weight".to_string(), GgufTensorInfo { dims: vec![256, 512, 4], ggml_type: 12 });
        infos.insert("blk.0.ffn_down_exps.weight".to_string(), GgufTensorInfo { dims: vec![512, 256, 4], ggml_type: 14 });

        let bytes = expert_vram_bytes_for_layer(&infos, 0, 4).expect("computes ok");
        assert_eq!(bytes, 254_976);
        // Rounds UP to 1 MB even though 254976 < 1_048_576 — never an
        // under-estimate, matching cs-miner's own `moe_expert_vram_mb` doc.
        assert_eq!(bytes.div_ceil(1024 * 1024), 1);
    }

    #[test]
    fn expert_vram_bytes_for_layer_errors_on_missing_tensor() {
        let infos = HashMap::new();
        let err = expert_vram_bytes_for_layer(&infos, 0, 4).unwrap_err();
        assert!(err.contains("blk.0.ffn_gate_exps.weight"));
    }

    #[test]
    fn expert_vram_bytes_for_layer_errors_on_unsupported_dtype() {
        let mut infos = HashMap::new();
        infos.insert("blk.0.ffn_gate_exps.weight".to_string(), GgufTensorInfo { dims: vec![256, 512, 4], ggml_type: 9999 });
        infos.insert("blk.0.ffn_up_exps.weight".to_string(), GgufTensorInfo { dims: vec![256, 512, 4], ggml_type: 12 });
        infos.insert("blk.0.ffn_down_exps.weight".to_string(), GgufTensorInfo { dims: vec![512, 256, 4], ggml_type: 14 });
        let err = expert_vram_bytes_for_layer(&infos, 0, 4).unwrap_err();
        assert!(err.contains("unsupported/unknown ggml_type 9999"));
    }

    #[test]
    fn expert_vram_bytes_for_layer_errors_on_zero_n_expert() {
        let infos = HashMap::new();
        assert!(expert_vram_bytes_for_layer(&infos, 0, 0).is_err());
    }

    /// End-to-end: a real (ephemeral, in-process) HTTP server serves a
    /// synthetic GGUF, `ExpertVramEstimator` fetches+parses+caches it, and
    /// answers `min_vram_mb_for_range` with the hand-computed expected MB —
    /// exercises the full fetch -> parse -> cache -> answer path with no
    /// mocking library (uses `axum`, already a dependency of this crate).
    #[tokio::test]
    async fn estimator_end_to_end_over_real_http() {
        let bytes = build_synthetic_gguf(3, 256, 512, 4, 12); // gate/up=Q4_K, down=Q4_K too (dtype param applies to all 3 here)
        let bytes = Arc::new(bytes);

        let app_bytes = bytes.clone();
        let app = axum::Router::new().route(
            "/model.gguf",
            axum::routing::get(move || {
                let bytes = app_bytes.clone();
                async move { (*bytes).clone() }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });

        let estimator = ExpertVramEstimator::new();
        let url = format!("http://{addr}/model.gguf");

        // n_expert=4, all three tensors Q4_K(12): per-expert elem_count=131072,
        // blocks=512, bytes=512*144=73728 per tensor, *3 = 221184 bytes -> 1 MB (rounded up).
        let mb = estimator
            .min_vram_mb_for_range("TEST_MODEL", &url, "sha-doesnt-matter-here", 0, 3, 4)
            .await
            .expect("estimate succeeds against the real HTTP server above");
        assert_eq!(mb, 1);

        // Second call (a different layer range) must be served entirely
        // from the cache populated by the first call — no second header
        // fetch/parse happens for the same (model_id, gguf_sha256) key.
        let mb_cached = estimator.min_vram_mb_for_range("TEST_MODEL", &url, "sha-doesnt-matter-here", 1, 2, 4).await.expect("cached estimate");
        assert_eq!(mb_cached, 1);
    }

    #[tokio::test]
    async fn estimator_returns_none_on_unreachable_host() {
        let estimator = ExpertVramEstimator::new();
        // Port 1 is reserved/unlikely to have anything listening — a fast,
        // deterministic connection failure without depending on any real
        // network condition.
        let result = estimator.min_vram_mb_for_range("TEST_MODEL", "http://127.0.0.1:1/model.gguf", "sha", 0, 4, 4).await;
        assert_eq!(result, None);
    }
}
