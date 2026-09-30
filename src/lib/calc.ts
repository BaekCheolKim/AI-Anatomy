/**
 * Every number AI-Flow displays comes from a function in this file (directly, or
 * via the tensor catalog, which uses `derive` from here). Functions return
 * `Derived` values that carry their formula id and the values substituted into
 * it, so the inspector can always show the derivation next to the number.
 *
 * Nothing here knows about a specific model: branches are on config features
 * (`attention.kind`, `ffn.kind`, `position.kind`) only.
 */
import type {
  AttentionConfig,
  Derived,
  Dtype,
  DtypeRef,
  FormulaDef,
  FormulaId,
  GpuSpec,
  MlpConfig,
  ModelConfig,
  ModelSpec,
  NormConfig,
  Recompute,
  Scenario,
} from '../types';

// ─────────────────────────────── dtype ───────────────────────────────

export const DTYPE_BYTES: Readonly<Record<Dtype, number>> = {
  int32: 4,
  fp32: 4,
  tf32: 4, // stored as 32-bit; only the tensor-core multiply uses a 10-bit mantissa
  bf16: 2,
  fp16: 2,
  fp8_e4m3: 1,
  int8: 1,
  int4: 0.5, // excludes per-group scales (≈ +0.125 bit/param at group size 128 with fp16 scales)
};

export function bytesPerParam(d: Dtype): number {
  return DTYPE_BYTES[d];
}

export function resolveDtype(ref: DtypeRef, s: Scenario): Dtype {
  if (typeof ref === 'string') return ref;
  switch (ref.role) {
    case 'weight':
      return s.weightDtype;
    case 'kv':
      return s.kvDtype;
    case 'act':
      return s.actDtype;
  }
}

// ───────────────────────────── Formulas ─────────────────────────────

const SRC = {
  kaplan: 'https://arxiv.org/abs/2001.08361',
  korthikanti: 'https://arxiv.org/abs/2205.05198',
  megatronPP: 'https://arxiv.org/abs/2104.04473',
  zero: 'https://arxiv.org/abs/1910.02054',
  roofline: 'https://doi.org/10.1145/1498765.1498785',
  ncclPerf: 'https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md',
  gqa: 'https://arxiv.org/abs/2305.13245',
  mla: 'https://arxiv.org/abs/2405.04434',
  flash2: 'https://arxiv.org/abs/2307.08691',
} as const;

export const FORMULAS: Readonly<Record<FormulaId, FormulaDef>> = {
  bytesPerParam: {
    name: 'Bytes per element',
    unit: 'bytes',
    tex: 'b(\\text{dtype})',
    approx: false,
  },
  tensorBytes: {
    name: 'Tensor size',
    unit: 'bytes',
    tex: '\\text{bytes} = \\prod_i d_i \\cdot b(\\text{dtype})',
    approx: false,
  },
  perRankBytes: {
    name: 'Per-rank share',
    unit: 'bytes',
    tex: '\\text{bytes}_{rank} = \\text{bytes} \\cdot \\prod_{\\text{shards}} \\frac{\\lceil d / w \\rceil}{d}',
    note: 'Each sharded dimension d is split w ways; ⌈d/w⌉ means over-splitting (w > d) leaves a whole element per rank (replication) rather than a fraction.',
    approx: false,
  },
  paramsTotal: {
    name: 'Total parameters',
    unit: 'params',
    tex: 'N = N_{emb} + N_{pos} + N_{attn} + N_{ffn} + N_{router} + N_{experts} + N_{norm} + N_{head}',
    note: 'Counted from the config matrix by matrix, not copied from the model card.',
    approx: false,
  },
  moeActiveParams: {
    name: 'Active parameters per token',
    unit: 'params',
    tex: 'N_{active} = N_{dense} + N_{routed} \\cdot \\frac{k}{E}',
    note: 'N_dense is every parameter outside the routed experts (attention, shared experts, router, embeddings).',
    approx: false,
  },
  matmulParams: {
    name: 'Matmul parameters per token',
    unit: 'params',
    tex: 'N_{mm} = N_{active} - N_{emb} - N_{pos} + [\\text{tied}] \\cdot V H',
    note: 'The N in 2N and 6N FLOP rules. Embedding lookups are gathers, not matmuls, so they cost no FLOPs; a tied LM head is still a V×H matmul.',
    approx: false,
    source: SRC.kaplan,
  },
  paramsPerRank: {
    name: 'Parameters held per GPU',
    unit: 'params',
    tex: 'N_{rank} = \\frac{N_{dense}}{tp \\cdot pp \\cdot z_d} + \\frac{N_{routed}}{pp \\cdot g \\cdot z_e}',
    note: 'g = ep if ep > 1, else tp (experts are TP-split when not expert-parallel). z = DP ways the ZeRO stage shards over (1 when inactive); for experts z_e = tp·dp/g, the expert-data-parallel group. Assumes even splits.',
    approx: false,
    source: SRC.zero,
  },
  weightBytes: {
    name: 'Weight memory',
    unit: 'bytes',
    tex: 'W = N \\cdot b_w',
    approx: false,
  },
  kvBytesGQA: {
    name: 'KV cache (MHA/GQA)',
    unit: 'bytes',
    tex: '\\text{KV} = 2 \\cdot L \\cdot N_{kv} \\cdot D \\cdot S \\cdot B \\cdot b_{kv}',
    note: '2 = one K and one V per layer, per KV head, per position. GQA shrinks N_kv below N_h; MHA has N_kv = N_h.',
    approx: false,
    source: SRC.gqa,
  },
  kvBytesMLA: {
    name: 'KV cache (MLA)',
    unit: 'bytes',
    tex: '\\text{KV} = L \\cdot (d_c + d_r) \\cdot S \\cdot B \\cdot b_{kv}',
    note: 'Only the shared latent c_KV (d_c = kv_lora_rank) and the decoupled RoPE key (d_r) are cached, once per position for all heads. No factor of 2 and no head count.',
    approx: false,
    source: SRC.mla,
  },
  fwdFlops: {
    name: 'Forward FLOPs (matmuls)',
    unit: 'flops',
    tex: 'F_{fwd} = 2 \\cdot N_{mm} \\cdot T',
    note: 'One multiply + one add per weight per token. Excludes the attention score term, shown separately.',
    approx: false,
    source: SRC.kaplan,
  },
  trainFlops: {
    name: 'Training FLOPs',
    unit: 'flops',
    tex: 'F_{train} = c \\cdot N_{mm} \\cdot T, \\quad c = 6 \\;(+2 \\text{ with full recompute})',
    note: 'Forward 2N + backward 4N (gradients w.r.t. both activations and weights). Full activation recompute re-runs the forward: 8N.',
    approx: false,
    source: SRC.kaplan,
  },
  attnFlops: {
    name: 'Attention score FLOPs',
    unit: 'flops',
    tex: 'F_{attn} = 2 \\cdot L \\cdot B \\cdot N_h \\cdot S_q \\cdot S_{kv} \\cdot (d_{qk} + d_v)',
    note: 'QKᵀ and softmax·V. With S_q = S_kv = S and d_qk = d_v = D this is 4·B·N_h·S²·D per layer: the quadratic term. Counts the full S×S matrix; causal kernels such as FlashAttention-2 skip ≈ half.',
    approx: false,
    source: SRC.flash2,
  },
  stepFlops: {
    name: 'Step FLOPs',
    unit: 'flops',
    tex: 'F = F_{fwd} + F_{attn}',
    approx: false,
  },
  savedActElems: {
    name: 'Saved activations per token per layer',
    unit: 'elements',
    tex: 'A = A_{attn} + A_{ffn}',
    note: 'Tensors kept from forward for backward, counted per sub-block (norm inputs, Q/K/V/O, FFN intermediates), FlashAttention assumed so no S² scores are stored. With full recompute only the block input (H) is kept. Excludes allocator temporaries.',
    approx: true,
    source: SRC.korthikanti,
  },
  workingSetElems: {
    name: 'Inference working set per token',
    unit: 'elements',
    tex: 'W_{act} = 2H + \\max(W_{attn}, W_{ffn})',
    note: 'Peak live activations while one layer runs (residual + normed input + the larger sub-block). Layers run one at a time, so this does not scale with L.',
    approx: true,
  },
  expertsTouched: {
    name: 'Distinct experts touched',
    unit: 'count',
    tex: 'E_t = E \\cdot \\left(1 - \\left(1 - \\tfrac{k}{E}\\right)^{T}\\right)',
    note: 'Expected distinct routed experts per MoE layer for T tokens under uniform routing. Real routers are skewed, so this is an upper-ish estimate.',
    approx: true,
  },
  weightBytesRead: {
    name: 'Weight bytes read per step',
    unit: 'bytes',
    tex: 'W_{read} = (N_{dense} + L_{moe} \\cdot E_t \\cdot N_x) \\cdot b_w',
    note: 'An MoE step only streams the experts some token was routed to. N_x = parameters per expert.',
    approx: true,
  },
  stepBytes: {
    name: 'HBM bytes per step',
    unit: 'bytes',
    tex: 'Q = W_{read} + \\text{KV}',
    note: 'Weights streamed once per step plus KV cache read (decode) or written (prefill). Activation traffic is ignored; it is small next to weights at decode.',
    approx: true,
  },
  arithmeticIntensity: {
    name: 'Arithmetic intensity',
    unit: 'flop/byte',
    tex: 'I = F / Q',
    approx: false,
    source: SRC.roofline,
  },
  decodeIntensity: {
    name: 'Decode arithmetic intensity',
    unit: 'flop/byte',
    tex: 'I_{dec} = \\frac{F_{fwd} + F_{attn}}{W_{read} + \\text{KV}_{read}}',
    note: 'Each decode step reads every weight and the whole KV cache to produce one token per sequence. Far below the GPU ridge point: memory-bandwidth-bound.',
    approx: false,
    source: SRC.roofline,
  },
  ridgePoint: {
    name: 'Ridge point',
    unit: 'flop/byte',
    tex: 'I^* = \\pi / \\beta',
    note: 'Intensity where a kernel stops being bandwidth-bound. π = peak dense FLOP/s for the compute dtype, β = HBM bandwidth.',
    approx: false,
    source: SRC.roofline,
  },
  rooflineTime: {
    name: 'Roofline time',
    unit: 'seconds',
    tex: 't = \\max\\left(\\frac{F}{\\pi}, \\frac{Q}{\\beta}\\right)',
    note: 'Lower bound: assumes peak throughput and perfect overlap of compute and memory traffic.',
    approx: true,
    source: SRC.roofline,
  },
  allReduceBytes: {
    name: 'All-reduce traffic per rank',
    unit: 'bytes',
    tex: 'V_{AR} = 2 \\cdot \\frac{n-1}{n} \\cdot M',
    note: 'Ring all-reduce = reduce-scatter + all-gather, each moving (n−1)/n of the tensor through every rank.',
    approx: false,
    source: SRC.ncclPerf,
  },
  reduceScatterBytes: {
    name: 'Reduce-scatter traffic per rank',
    unit: 'bytes',
    tex: 'V_{RS} = \\frac{n-1}{n} \\cdot M',
    approx: false,
    source: SRC.ncclPerf,
  },
  allGatherBytes: {
    name: 'All-gather traffic per rank',
    unit: 'bytes',
    tex: 'V_{AG} = \\frac{n-1}{n} \\cdot M',
    approx: false,
    source: SRC.ncclPerf,
  },
  allToAllBytes: {
    name: 'All-to-all traffic per rank',
    unit: 'bytes',
    tex: 'V_{A2A} = \\frac{n-1}{n} \\cdot M',
    note: 'Each rank keeps 1/n of its buffer and sends the rest; used for MoE token dispatch and combine.',
    approx: false,
    source: SRC.ncclPerf,
  },
  tpShardedHidden: {
    name: 'TP shard width',
    unit: 'count',
    tex: 'H_{tp} = H / tp',
    note: 'Width each rank owns of a column-parallel projection. The residual stream itself stays full width H on every TP rank.',
    approx: false,
    source: SRC.megatronPP,
  },
  ppBubbleFraction: {
    name: 'Pipeline bubble',
    unit: 'ratio',
    tex: '\\text{bubble} = \\frac{pp - 1}{m}',
    note: 'Idle time relative to ideal compute time for GPipe / 1F1B schedules with m micro-batches. More micro-batches amortise the fill and drain.',
    approx: false,
    source: SRC.megatronPP,
  },
  bucketSum: {
    name: 'Budget bucket',
    unit: 'bytes',
    tex: '\\sum_{t \\in \\text{bucket}} \\text{bytes}_{rank}(t)',
    approx: false,
  },
  runtimeOverhead: {
    name: 'Runtime + fragmentation',
    unit: 'bytes',
    tex: 'R = C_{ctx} + f \\cdot U',
    note: 'C_ctx ≈ 1 GiB for the CUDA context, cuBLAS/NCCL workspaces; f ≈ 5% allocator fragmentation of the dynamic allocations U. Both vary by framework and version.',
    approx: true,
  },
  hbmUsed: {
    name: 'HBM used per GPU',
    unit: 'bytes',
    tex: '\\sum \\text{buckets}',
    approx: false,
  },
  hbmCapacity: {
    name: 'HBM capacity',
    unit: 'bytes',
    tex: 'C_{HBM}',
    note: 'Nominal capacity from the datasheet; nvidia-smi reports slightly less usable memory.',
    approx: false,
  },
};

type Terms = Readonly<Record<string, number | Derived>>;

/** The only constructor of `Derived`. Approximation propagates from the formula and from any nested term. */
export function derive(formula: FormulaId, value: number, terms: Terms, approx = false): Derived {
  const def = FORMULAS[formula];
  const inherited = Object.values(terms).some((t) => typeof t !== 'number' && t.approx);
  return { value, unit: def.unit, formula, terms, approx: approx || def.approx || inherited } as Derived;
}

export function num(t: number | Derived): number {
  return typeof t === 'number' ? t : t.value;
}

// ─────────────────────── Model structure (plain numbers) ───────────────────────

export interface AttnDims {
  heads: number;
  /** Cached K/V heads; null for MLA, which caches a shared latent instead. */
  kvHeads: number | null;
  qkHeadDim: number;
  vHeadDim: number;
}

export function attnDims(a: AttentionConfig): AttnDims {
  switch (a.kind) {
    case 'MHA':
      return { heads: a.heads, kvHeads: a.heads, qkHeadDim: a.headDim, vHeadDim: a.headDim };
    case 'GQA':
      return { heads: a.heads, kvHeads: a.kvHeads, qkHeadDim: a.headDim, vHeadDim: a.headDim };
    case 'MLA':
      return { heads: a.heads, kvHeads: null, qkHeadDim: a.qkNopeHeadDim + a.qkRopeHeadDim, vHeadDim: a.vHeadDim };
  }
}

export interface LayerCounts {
  dense: number;
  moe: number;
}

export function layerCounts(cfg: ModelConfig): LayerCounts {
  if (cfg.ffn.kind === 'dense') return { dense: cfg.layers, moe: 0 };
  const dense = cfg.ffn.leadingDense?.layers ?? 0;
  return { dense, moe: cfg.layers - dense };
}

/** FFN used by dense layers, if the model has any. */
export function denseMlp(cfg: ModelConfig): MlpConfig | null {
  return cfg.ffn.kind === 'dense' ? cfg.ffn.mlp : (cfg.ffn.leadingDense?.mlp ?? null);
}

function attentionParamsPerLayer(a: AttentionConfig, H: number): number {
  switch (a.kind) {
    case 'MHA':
    case 'GQA': {
      const kv = a.kind === 'MHA' ? a.heads : a.kvHeads;
      const q = a.heads * a.headDim;
      const k = kv * a.headDim;
      const weights = H * q + 2 * H * k + q * H;
      return weights + (a.bias ? q + 2 * k + H : 0);
    }
    case 'MLA': {
      const dqk = a.qkNopeHeadDim + a.qkRopeHeadDim;
      const qProj =
        a.qLoraRank === null
          ? H * a.heads * dqk
          : H * a.qLoraRank + a.qLoraRank /* q_a_layernorm */ + a.qLoraRank * a.heads * dqk;
      const kvProj =
        H * (a.kvLoraRank + a.qkRopeHeadDim) +
        a.kvLoraRank /* kv_a_layernorm */ +
        a.kvLoraRank * a.heads * (a.qkNopeHeadDim + a.vHeadDim);
      const oProj = a.heads * a.vHeadDim * H;
      return qProj + kvProj + oProj;
    }
  }
}

function mlpParams(mlp: MlpConfig, H: number): number {
  const F = mlp.interDim;
  return mlp.gated ? 3 * H * F + (mlp.bias ? 2 * F + H : 0) : 2 * H * F + (mlp.bias ? F + H : 0);
}

function normParams(n: NormConfig, H: number): number {
  return n.kind === 'LayerNorm' ? 2 * H : H;
}

export interface ParamBreakdown {
  embedding: number;
  positional: number;
  attention: number;
  norms: number;
  ffnDense: number;
  router: number;
  expertsShared: number;
  expertsRouted: number;
  /** 0 when tied to the embedding. */
  lmHead: number;
  total: number;
  /** Everything outside the routed experts. */
  dense: number;
  /** Touched per token. */
  active: number;
  /** In matmuls per token: the N of 2N / 6N. */
  matmul: number;
  /** One expert (routed or shared). 0 for dense models. */
  perExpert: number;
}

const breakdownCache = new WeakMap<ModelConfig, ParamBreakdown>();

export function paramBreakdown(cfg: ModelConfig): ParamBreakdown {
  const hit = breakdownCache.get(cfg);
  if (hit) return hit;

  const { hidden: H, vocab: V, layers: L } = cfg;
  const lc = layerCounts(cfg);
  const dm = denseMlp(cfg);
  const embedding = V * H;
  const positional = cfg.position.kind === 'learned' ? cfg.maxPositions * H : 0;
  const attention = L * attentionParamsPerLayer(cfg.attention, H);
  const norms = (2 * L + 1) * normParams(cfg.norm, H);
  const ffnDense = dm ? lc.dense * mlpParams(dm, H) : 0;

  let router = 0;
  let expertsShared = 0;
  let expertsRouted = 0;
  let perExpert = 0;
  let activeRouted = 0;
  if (cfg.ffn.kind === 'moe') {
    const f = cfg.ffn;
    perExpert = mlpParams(f.expert, H);
    router = lc.moe * (f.routedExperts * H + (f.router.selectionBias ? f.routedExperts : 0));
    expertsShared = lc.moe * f.sharedExperts * perExpert;
    expertsRouted = lc.moe * f.routedExperts * perExpert;
    activeRouted = lc.moe * f.topK * perExpert;
  }

  const lmHead = cfg.tiedEmbeddings ? 0 : V * H;
  const total = embedding + positional + attention + norms + ffnDense + router + expertsShared + expertsRouted + lmHead;
  const dense = total - expertsRouted;
  const active = dense + activeRouted;
  const matmul = active - embedding - positional + (cfg.tiedEmbeddings ? V * H : 0);

  const out: ParamBreakdown = {
    embedding,
    positional,
    attention,
    norms,
    ffnDense,
    router,
    expertsShared,
    expertsRouted,
    lmHead,
    total,
    dense,
    active,
    matmul,
    perExpert,
  };
  breakdownCache.set(cfg, out);
  return out;
}

// ─────────────────────────────── Parameters ───────────────────────────────

function nonZero(terms: Record<string, number>): Record<string, number> {
  return Object.fromEntries(Object.entries(terms).filter(([, v]) => v !== 0));
}

export function paramsTotal(m: ModelSpec): Derived {
  const p = paramBreakdown(m.config);
  return derive(
    'paramsTotal',
    p.total,
    nonZero({
      N_emb: p.embedding,
      N_pos: p.positional,
      N_attn: p.attention,
      N_ffn: p.ffnDense,
      N_router: p.router,
      N_shared: p.expertsShared,
      N_routed: p.expertsRouted,
      N_norm: p.norms,
      N_head: p.lmHead,
    }),
  );
}

/** Parameters touched per token. Dense models: equal to the total. */
export function activeParams(m: ModelSpec): Derived {
  if (m.config.ffn.kind === 'dense') return paramsTotal(m);
  const p = paramBreakdown(m.config);
  const f = m.config.ffn;
  return derive('moeActiveParams', p.active, {
    N_dense: p.dense,
    N_routed: p.expertsRouted,
    k: f.topK,
    E: f.routedExperts,
  });
}

export function matmulParams(m: ModelSpec): Derived {
  const p = paramBreakdown(m.config);
  const terms: Record<string, number | Derived> = { N_active: activeParams(m), N_emb: p.embedding };
  if (p.positional) terms.N_pos = p.positional;
  if (m.config.tiedEmbeddings) terms.VH = m.config.vocab * m.config.hidden;
  return derive('matmulParams', p.matmul, terms);
}

function zeroActive(s: Scenario, atLeast: 1 | 2 | 3): boolean {
  return s.workload === 'training' && s.zeroStage >= atLeast;
}

/** Expert-parallel group: experts are spread over ep ranks, or TP-split when ep = 1. */
export function expertGroup(s: Scenario): number {
  return s.ep > 1 ? s.ep : s.tp;
}

/** Replicas of each expert shard, i.e. the group expert gradients are reduced / ZeRO-sharded over. */
export function expertDataParallel(s: Scenario): number {
  return (s.tp * s.dp) / expertGroup(s);
}

/**
 * Parameters resident on one GPU. `zeroAtLeast` picks which ZeRO stage shards
 * this copy over DP: 3 for weights, 2 for gradients, 1 for optimizer state.
 */
export function paramsPerRank(m: ModelSpec, s: Scenario, zeroAtLeast: 1 | 2 | 3): Derived {
  const p = paramBreakdown(m.config);
  const z = zeroActive(s, zeroAtLeast);
  const zd = z ? s.dp : 1;
  const g = expertGroup(s);
  const ze = z ? expertDataParallel(s) : 1;
  const value = p.dense / (s.tp * s.pp * zd) + (p.expertsRouted ? p.expertsRouted / (s.pp * g * ze) : 0);
  const terms: Record<string, number> = { N_dense: p.dense, tp: s.tp, pp: s.pp, z_d: zd };
  if (p.expertsRouted) Object.assign(terms, { N_routed: p.expertsRouted, g, z_e: ze });
  return derive('paramsPerRank', value, terms);
}

// ─────────────────────────────── Memory ───────────────────────────────

/** W = N · b_w. Pass a per-rank N (from `paramsPerRank`) for the per-GPU share. */
export function weightBytes(m: ModelSpec, s: Scenario, params: Derived = paramsTotal(m)): Derived {
  const bw = bytesPerParam(s.weightDtype);
  return derive('weightBytes', params.value * bw, { N: params, b_w: bw });
}

export interface KvOptions {
  /** Sequences; default the scenario batch. */
  batch?: number;
  /** Cached positions per sequence; default seqLen + genTokens (end of generation). */
  tokens?: number;
}

/** KV cache for the whole model. Dispatches on the attention feature, not on the model. */
export function kvBytes(m: ModelSpec, s: Scenario, opts: KvOptions = {}): Derived {
  const B = opts.batch ?? s.batch;
  const S = opts.tokens ?? s.seqLen + s.genTokens;
  const L = m.config.layers;
  const b = bytesPerParam(s.kvDtype);
  const a = m.config.attention;
  switch (a.kind) {
    case 'MHA':
    case 'GQA': {
      const Nkv = a.kind === 'MHA' ? a.heads : a.kvHeads;
      return derive('kvBytesGQA', 2 * L * Nkv * a.headDim * S * B * b, {
        L,
        N_kv: Nkv,
        D: a.headDim,
        S,
        B,
        b_kv: b,
      });
    }
    case 'MLA':
      return derive('kvBytesMLA', L * (a.kvLoraRank + a.qkRopeHeadDim) * S * B * b, {
        L,
        d_c: a.kvLoraRank,
        d_r: a.qkRopeHeadDim,
        S,
        B,
        b_kv: b,
      });
  }
}

function attnSubBlockElems(cfg: ModelConfig): number {
  const a = cfg.attention;
  const d = attnDims(a);
  switch (a.kind) {
    case 'MHA':
    case 'GQA':
      // Q, K, V, O
      return 2 * d.heads * d.qkHeadDim + 2 * (d.kvHeads ?? d.heads) * d.qkHeadDim;
    case 'MLA':
      // q latent, Q, kv latent ‖ k_rope, K and V up-projected, O
      return (a.qLoraRank ?? 0) + 2 * d.heads * d.qkHeadDim + (a.kvLoraRank + a.qkRopeHeadDim) + 2 * d.heads * d.vHeadDim;
  }
}

function ffnSubBlockElems(cfg: ModelConfig, layer: 'dense' | 'moe', perMatrixCopies: { gated: number; plain: number }): number {
  const H = cfg.hidden;
  if (layer === 'dense') {
    const mlp = denseMlp(cfg);
    if (!mlp) return 0;
    return (mlp.gated ? perMatrixCopies.gated : perMatrixCopies.plain) * mlp.interDim;
  }
  if (cfg.ffn.kind !== 'moe') return 0;
  const f = cfg.ffn;
  const perExpert = (f.expert.gated ? perMatrixCopies.gated : perMatrixCopies.plain) * f.expert.interDim;
  // router logits + expert intermediates + dispatched-in / pre-combine-out token copies
  return f.routedExperts + (f.topK + f.sharedExperts) * perExpert + 2 * f.topK * H;
}

/** Elements saved for backward per token per layer, averaged over layer types. */
export function savedActElems(m: ModelSpec, recompute: Recompute): Derived {
  const cfg = m.config;
  const H = cfg.hidden;
  if (recompute === 'full') return derive('savedActElems', H, { H });
  const lc = layerCounts(cfg);
  const attn = 2 * H + attnSubBlockElems(cfg);
  // gated: gate, up, act(gate), gate⊙up; plain: fc1 out, act out
  const copies = { gated: 4, plain: 2 };
  const ffnDense = 2 * H + ffnSubBlockElems(cfg, 'dense', copies);
  const ffnMoe = 2 * H + ffnSubBlockElems(cfg, 'moe', copies);
  const ffnAvg = (lc.dense * ffnDense + lc.moe * ffnMoe) / cfg.layers;
  const terms: Record<string, number> = { A_attn: attn };
  if (lc.dense) terms.A_ffn_dense = ffnDense;
  if (lc.moe) terms.A_ffn_moe = ffnMoe;
  if (lc.dense && lc.moe) Object.assign(terms, { L_dense: lc.dense, L_moe: lc.moe });
  return derive('savedActElems', attn + ffnAvg, terms);
}

/** Peak live activation elements per token while one layer runs (inference). */
export function workingSetElems(m: ModelSpec): Derived {
  const cfg = m.config;
  const H = cfg.hidden;
  const lc = layerCounts(cfg);
  const attn = attnSubBlockElems(cfg);
  // gated: gate, up, gate⊙up (activation fused in place); plain: fc1 out, act out
  const copies = { gated: 3, plain: 2 };
  const ffn = Math.max(
    lc.dense ? ffnSubBlockElems(cfg, 'dense', copies) : 0,
    lc.moe ? ffnSubBlockElems(cfg, 'moe', copies) : 0,
  );
  return derive('workingSetElems', 2 * H + Math.max(attn, ffn), { H, W_attn: attn, W_ffn: ffn });
}

// ─────────────────────────────── FLOPs ───────────────────────────────

export function fwdFlops(m: ModelSpec, tokens: number): Derived {
  const N = matmulParams(m);
  return derive('fwdFlops', 2 * N.value * tokens, { N_mm: N, T: tokens });
}

export function trainFlops(m: ModelSpec, tokens: number, recompute: Recompute, approxTokens = false): Derived {
  const N = matmulParams(m);
  const c = recompute === 'full' ? 8 : 6;
  return derive('trainFlops', c * N.value * tokens, { c, N_mm: N, T: tokens }, approxTokens);
}

export function attnFlops(m: ModelSpec, batch: number, qLen: number, kvLen: number): Derived {
  const d = attnDims(m.config.attention);
  const L = m.config.layers;
  return derive('attnFlops', 2 * L * batch * d.heads * qLen * kvLen * (d.qkHeadDim + d.vHeadDim), {
    L,
    B: batch,
    N_h: d.heads,
    S_q: qLen,
    S_kv: kvLen,
    d_qk: d.qkHeadDim,
    d_v: d.vHeadDim,
  });
}

/** Expected distinct routed experts per MoE layer for `tokens` tokens. Null for dense models. */
export function expertsTouched(m: ModelSpec, tokens: number): Derived | null {
  if (m.config.ffn.kind !== 'moe') return null;
  const { routedExperts: E, topK: k } = m.config.ffn;
  return derive('expertsTouched', E * (1 - Math.pow(1 - k / E, tokens)), { E, k, T: tokens });
}

/** Weight bytes streamed from HBM by one forward step over `tokens` tokens. */
export function weightBytesRead(m: ModelSpec, s: Scenario, tokens: number): Derived {
  const touched = expertsTouched(m, tokens);
  if (!touched) return weightBytes(m, s);
  const p = paramBreakdown(m.config);
  const Lm = layerCounts(m.config).moe;
  const bw = bytesPerParam(s.weightDtype);
  return derive('weightBytesRead', (p.dense + Lm * touched.value * p.perExpert) * bw, {
    N_dense: p.dense,
    L_moe: Lm,
    E_t: touched,
    N_x: p.perExpert,
    b_w: bw,
  });
}

// ─────────────────────────── Roofline / steps ───────────────────────────

export interface StepProfile {
  flops: Derived;
  bytes: Derived;
  intensity: Derived;
}

/** Sequences per model replica. */
export function replicaBatch(s: Scenario): number {
  return Math.ceil(s.batch / s.dp);
}

/** One decode step for one model replica, at the last generated position (largest KV read). */
export function decodeStep(m: ModelSpec, s: Scenario): StepProfile {
  const b = replicaBatch(s);
  const ctx = s.seqLen + s.genTokens;
  const fMm = fwdFlops(m, b);
  const fAttn = attnFlops(m, b, 1, ctx);
  const flops = derive('stepFlops', fMm.value + fAttn.value, { F_fwd: fMm, F_attn: fAttn });
  const w = weightBytesRead(m, s, b);
  const kv = kvBytes(m, s, { batch: b, tokens: ctx });
  const bytes = derive('stepBytes', w.value + kv.value, { W_read: w, KV_read: kv });
  const intensity = derive('decodeIntensity', flops.value / bytes.value, { F: flops, Q: bytes });
  return { flops, bytes, intensity };
}

/** Prefill of the whole prompt for one model replica. */
export function prefillStep(m: ModelSpec, s: Scenario): StepProfile {
  const b = replicaBatch(s);
  const S = s.seqLen;
  const fMm = fwdFlops(m, b * S);
  const fAttn = attnFlops(m, b, S, S);
  const flops = derive('stepFlops', fMm.value + fAttn.value, { F_fwd: fMm, F_attn: fAttn });
  const w = weightBytesRead(m, s, b * S);
  const kv = kvBytes(m, s, { batch: b, tokens: S });
  const bytes = derive('stepBytes', w.value + kv.value, { W_read: w, KV_write: kv });
  const intensity = derive('arithmeticIntensity', flops.value / bytes.value, { F: flops, Q: bytes });
  return { flops, bytes, intensity };
}

/** Dense peak for a compute dtype; undefined when the GPU has no tensor-core path for it. */
export function peakFlops(gpu: GpuSpec, dtype: Dtype): number | undefined {
  return gpu.peakFlops[dtype];
}

export function ridgePoint(gpu: GpuSpec, dtype: Dtype): Derived | null {
  const pi = peakFlops(gpu, dtype);
  if (pi === undefined) return null;
  const beta = gpu.memory.bandwidth;
  return derive('ridgePoint', pi / beta, { π: pi, β: beta });
}

export function rooflineTime(flops: Derived, bytes: Derived, pi: number, beta: number): Derived {
  return derive('rooflineTime', Math.max(flops.value / pi, bytes.value / beta), { F: flops, Q: bytes, π: pi, β: beta });
}

// ───────────────────────────── Collectives ─────────────────────────────

export function allReduceBytes(M: number | Derived, n: number): Derived {
  return derive('allReduceBytes', n <= 1 ? 0 : (2 * (n - 1) * num(M)) / n, { n, M });
}

export function reduceScatterBytes(M: number | Derived, n: number): Derived {
  return derive('reduceScatterBytes', n <= 1 ? 0 : ((n - 1) * num(M)) / n, { n, M });
}

export function allGatherBytes(M: number | Derived, n: number): Derived {
  return derive('allGatherBytes', n <= 1 ? 0 : ((n - 1) * num(M)) / n, { n, M });
}

export function allToAllBytes(M: number | Derived, n: number): Derived {
  return derive('allToAllBytes', n <= 1 ? 0 : ((n - 1) * num(M)) / n, { n, M });
}

// ───────────────────────────── Parallelism ─────────────────────────────

export function tpShardedHidden(m: ModelSpec, s: Scenario): Derived {
  return derive('tpShardedHidden', m.config.hidden / s.tp, { H: m.config.hidden, tp: s.tp });
}

export function ppBubbleFraction(s: Scenario): Derived {
  return derive('ppBubbleFraction', (s.pp - 1) / s.microBatches, { pp: s.pp, m: s.microBatches });
}
