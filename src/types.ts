/**
 * AI-Flow core schema.
 *
 * The root object is the MODEL, not the diagram:
 *
 *   render(model, lens, depth, scenario) -> graph + explanation
 *
 * - The graph STRUCTURE is a pure function of (model, lens): shared lens templates
 *   read `ModelSpec.config` and branch on architectural *features*
 *   (`attention.kind`, `ffn.kind`, `position.kind`), never on a model id.
 * - Depth only filters that graph (`minDepth`), so expanding never navigates away.
 * - Every NUMBER on the graph is a pure function of (model, scenario) and is
 *   carried as a `Derived`, which records the formula it came from.
 *
 * Nothing in this file knows about any particular model.
 */

// ───────────────────────────── Enumerations ─────────────────────────────
// Runtime tuples so URL parsing can validate input against the same source.

export const DTYPES = ['int32', 'fp32', 'tf32', 'bf16', 'fp16', 'fp8_e4m3', 'int8', 'int4'] as const;
export type Dtype = (typeof DTYPES)[number];

export const LENS_IDS = ['architecture', 'inference', 'training', 'memory', 'parallelism', 'hpc'] as const;
export type LensId = (typeof LENS_IDS)[number];

export const PARALLEL_AXES = ['tp', 'pp', 'dp', 'ep'] as const;
export type ParallelAxis = (typeof PARALLEL_AXES)[number];

export const DEPTHS = [1, 2, 3] as const;
export type Depth = (typeof DEPTHS)[number];

export type MemLocation = 'CPU RAM' | 'GPU HBM' | 'GPU SRAM' | 'NVMe' | 'Interconnect';
export type NodeKind = 'data' | 'compute' | 'memory' | 'hardware' | 'collective';
export type Workload = 'inference' | 'training';
/** Activation checkpointing. 'full' keeps only each block's input and recomputes the rest in backward. */
export type Recompute = 'none' | 'full';
export type ZeroStage = 0 | 1 | 2 | 3;

// ─────────────────────────────── Scenario ───────────────────────────────

/**
 * Global, live "what are we running" controls. Changing any field re-derives
 * every number in every lens; it never changes graph structure.
 */
export interface Scenario {
  workload: Workload;
  /** Sequences in flight across the whole deployment (global batch). */
  batch: number;
  /** Prompt length (inference) or training sequence length, in tokens. */
  seqLen: number;
  /** Tokens generated per sequence after the prompt (inference only). */
  genTokens: number;
  weightDtype: Dtype;
  kvDtype: Dtype;
  /** Activation dtype; also the matmul compute dtype (weights are dequantized to it). */
  actDtype: Dtype;
  gpuCount: number;
  gpuId: string;
  /** Parallel degrees. tp·pp·dp must equal gpuCount; ep partitions experts within tp×dp. */
  tp: number;
  pp: number;
  dp: number;
  ep: number;
  /** Micro-batches per optimizer step (pipeline micro-batches / gradient-accumulation steps). */
  microBatches: number;
  recompute: Recompute;
  /** ZeRO / FSDP partitioning of optimizer (≥1), gradients (≥2), weights (3) over DP. Training only. */
  zeroStage: ZeroStage;
}

export interface ScenarioIssue {
  level: 'error' | 'warn' | 'info';
  field: keyof Scenario;
  message: string;
}

// ────────────────────────────── Model config ──────────────────────────────
// Discriminated unions rather than flat optional fields: an MLA model cannot
// exist without MLA dimensions, and every template `switch` on `kind` is
// exhaustiveness-checked, so a new variant is a compile error everywhere it
// needs handling instead of a silent fallthrough.

export type Activation = 'gelu_tanh' | 'silu';

interface AttentionBase {
  /** Query heads. */
  heads: number;
}

/** Multi-head attention: one K/V head per query head. */
export interface MhaConfig extends AttentionBase {
  kind: 'MHA';
  headDim: number;
  /** q/k/v/o projections carry bias vectors (GPT-2: yes). */
  bias: boolean;
}

/** Grouped-query attention: `heads / kvHeads` query heads share each K/V head. */
export interface GqaConfig extends AttentionBase {
  kind: 'GQA';
  kvHeads: number;
  headDim: number;
  bias: boolean;
}

/**
 * Multi-head latent attention (DeepSeek-V2/V3). K and V are reconstructed from a
 * shared low-rank latent `c_KV` (kvLoraRank), plus a small decoupled RoPE key
 * (qkRopeHeadDim) shared across heads. Only `c_KV ‖ k_rope` is cached.
 */
export interface MlaConfig extends AttentionBase {
  kind: 'MLA';
  /** Rank of the query compression; null = queries projected directly from the hidden state. */
  qLoraRank: number | null;
  kvLoraRank: number;
  qkNopeHeadDim: number;
  qkRopeHeadDim: number;
  vHeadDim: number;
}

export type AttentionConfig = MhaConfig | GqaConfig | MlaConfig;

export interface MlpConfig {
  interDim: number;
  /** true: W_down·(act(W_gate·x) ⊙ W_up·x)  (SwiGLU).  false: W_2·act(W_1·x). */
  gated: boolean;
  activation: Activation;
  bias: boolean;
}

export interface DenseFfn {
  kind: 'dense';
  mlp: MlpConfig;
}

export interface RouterConfig {
  scoring: 'softmax' | 'sigmoid';
  /** Group-limited routing: experts are split into `groups`, tokens pick `topkGroups` groups first. */
  groups: number | null;
  topkGroups: number | null;
  /** Learned per-expert bias used only for selection (aux-loss-free balancing): E extra params per layer. */
  selectionBias: boolean;
}

export interface MoeFfn {
  kind: 'moe';
  /** One expert. Routed and shared experts have this shape. */
  expert: MlpConfig;
  routedExperts: number;
  sharedExperts: number;
  topK: number;
  router: RouterConfig;
  /** Leading layers that keep a dense FFN (DeepSeek `first_k_dense_replace`); null = every layer is MoE. */
  leadingDense: { layers: number; mlp: MlpConfig } | null;
}

export type FfnConfig = DenseFfn | MoeFfn;

export type RopeScaling =
  | { kind: 'llama3'; factor: number; lowFreqFactor: number; highFreqFactor: number; originalMaxPositions: number }
  | {
      kind: 'yarn';
      factor: number;
      originalMaxPositions: number;
      betaFast: number;
      betaSlow: number;
      mscale: number;
      mscaleAllDim: number;
    };

export type PositionConfig =
  /** Learned absolute position table with `maxPositions` rows, added to token embeddings. */
  | { kind: 'learned' }
  /** Rotary embeddings applied to q/k inside attention; no parameters. */
  | { kind: 'rope'; theta: number; scaling: RopeScaling | null };

export interface NormConfig {
  /** LayerNorm has γ and β (2H params); RMSNorm has γ only (H). Pre-norm placement is assumed. */
  kind: 'LayerNorm' | 'RMSNorm';
  eps: number;
}

export interface ModelConfig {
  layers: number;
  hidden: number;
  /** Embedding rows. May exceed the tokenizer's vocabulary when padded. */
  vocab: number;
  /** Context window in tokens (also the learned position table size). */
  maxPositions: number;
  attention: AttentionConfig;
  ffn: FfnConfig;
  position: PositionConfig;
  norm: NormConfig;
  /** LM head reuses the token embedding matrix. */
  tiedEmbeddings: boolean;
  /** Dtype of the published checkpoint. */
  nativeDtype: Dtype;
}

// ─────────────────────────────── Tensors ───────────────────────────────

/**
 * A tensor's dtype is either fixed (token ids are int32, Adam moments fp32) or
 * follows a scenario precision control, so switching BF16→FP8 re-sizes it.
 */
export type DtypeRole = 'weight' | 'kv' | 'act';
export type DtypeRef = Dtype | { role: DtypeRole };

export type Lifetime =
  | 'persistent' //     whole deployment / run (weights, optimizer state)
  | 'per-request' //    one sequence's lifetime (KV cache)
  | 'per-step' //       one optimizer step (gradients)
  | 'per-microbatch' // forward → backward of one micro-batch (saved activations)
  | 'per-token' //      one decode step
  | 'transient'; //     inside one op or fused kernel

/** Memory-lens budget bucket this tensor is counted in. Unbucketed tensors are views, not allocations. */
export type MemoryBucket = 'weights' | 'kv-cache' | 'activations' | 'gradients' | 'optimizer';

/**
 * One axis of partitioning: the shape dimension `dim` is split `ways` ways
 * (default: the scenario degree of `axis`). Per-rank size multiplies by
 * ⌈d/w⌉ / d, so over-splitting (tp > kvHeads) correctly leaves one head per rank.
 */
export interface Shard {
  axis: ParallelAxis;
  /** A dimension expression that appears verbatim in the tensor's `shape`. */
  dim: string;
  /** Dimension expression for the number of ways; defaults to the degree of `axis`. */
  ways?: string;
  /** Applies only when this ZeRO stage or higher is active (training only). */
  zeroAtLeast?: 1 | 2 | 3;
  /** Applies only when the degree of this axis is 1 (e.g. TP-shard experts only without EP). */
  onlyIfOne?: ParallelAxis;
}

/**
 * What a tensor IS, independent of which node shows it. Defined once per model
 * in the tensor catalog and referenced by id from nodes and edges in every lens,
 * so the KV cache in Inference and in Memory is literally the same object.
 *
 * Changed from the brief: shape and bytes are not separate callbacks. `shape` is a
 * list of dimension expressions (`'B'`, `'S+G'`, `'Nkv*D'`) evaluated against
 * the model+scenario, and bytes = numel × bytesPerParam(dtype). The symbolic and
 * concrete shapes cannot drift apart, and the inspector can print the derivation.
 */
export interface TensorSpec {
  id: string;
  label: string;
  dtype: DtypeRef;
  /** Logical (unsharded) shape, one dimension expression per axis. */
  shape: readonly string[];
  location: MemLocation;
  /** Absent/empty: every rank that touches it holds a full copy. */
  sharding?: readonly Shard[];
  /** Ownership nuance the shard list cannot express. */
  ownerNote?: string;
  lifetime: Lifetime;
  /** How it moves: who writes it, who reads it, over which link. */
  movement: string;
  bucket?: MemoryBucket;
  /** Exists only for these workloads. Absent: both. */
  workloads?: readonly Workload[];
  /**
   * The named, pedagogical formula for this tensor's size (e.g. kvBytesGQA). The
   * inspector shows it; tests assert it equals numel × bytes for every scenario.
   */
  explain?: (m: ModelSpec, s: Scenario) => Derived;
  note?: string;
}

// ─────────────────────────── Derived quantities ───────────────────────────

export type Unit =
  | 'bytes'
  | 'bytes/s'
  | 'flops'
  | 'flop/s'
  | 'flop/byte'
  | 'params'
  | 'elements'
  | 'count'
  | 'ratio'
  | 'seconds';

/** Every formula the UI can display. Definitions (KaTeX, caveats) live in `FORMULAS` in lib/calc.ts. */
export type FormulaId =
  | 'bytesPerParam'
  | 'tensorBytes'
  | 'perRankBytes'
  | 'paramsTotal'
  | 'moeActiveParams'
  | 'matmulParams'
  | 'paramsPerRank'
  | 'weightBytes'
  | 'kvBytesGQA'
  | 'kvBytesMLA'
  | 'fwdFlops'
  | 'trainFlops'
  | 'attnFlops'
  | 'stepFlops'
  | 'savedActElems'
  | 'workingSetElems'
  | 'expertsTouched'
  | 'weightBytesRead'
  | 'stepBytes'
  | 'arithmeticIntensity'
  | 'decodeIntensity'
  | 'ridgePoint'
  | 'rooflineTime'
  | 'allReduceBytes'
  | 'reduceScatterBytes'
  | 'allGatherBytes'
  | 'allToAllBytes'
  | 'tpShardedHidden'
  | 'ppBubbleFraction'
  | 'bucketSum'
  | 'runtimeOverhead'
  | 'hbmUsed'
  | 'hbmCapacity';

declare const derivedBrand: unique symbol;

/**
 * A number the UI is allowed to display. Only `derive()` in lib/calc.ts creates
 * these (the brand makes a bare literal a type error), so every rendered number
 * carries its formula and the values substituted into it.
 *
 * `terms` values are either plain inputs (config / scenario values) or nested
 * `Derived`s, so the inspector can expand a number into its full derivation tree.
 */
export interface Derived {
  readonly [derivedBrand]: true;
  readonly value: number;
  readonly unit: Unit;
  readonly formula: FormulaId;
  readonly terms: Readonly<Record<string, number | Derived>>;
  /** Estimate (formula is an approximation, or an input is flagged approximate). */
  readonly approx: boolean;
}

export interface FormulaDef {
  name: string;
  unit: Unit;
  /** Symbolic form, KaTeX. */
  tex: string;
  /** Assumptions and caveats shown next to the number. */
  note?: string;
  /** The formula itself is an estimate, not exact accounting. */
  approx: boolean;
  source?: string;
}

export interface Metric {
  label: string;
  value: Derived;
}

// ──────────────────────────────── Graph ────────────────────────────────

export interface NodeCost {
  flops?: Derived;
  bytesMoved?: Derived;
}

export interface FlowNode {
  id: string;
  label: string;
  kind: NodeKind;
  /** Appears at this depth and above. */
  minDepth: Depth;
  /** Expands out of this node: at depth ≥ minDepth the parent renders as a frame around its children. */
  parent?: string;
  /** Dimension expression: this node stands for N identical copies (`'L'` decoder blocks → "×32"). */
  repeat?: string;
  /** One line, shown on hover. */
  summary: string;
  /** Markdown, shown in the inspector. Prose only; live numbers come from tensors, cost and metrics. */
  detail: string;
  /** KaTeX. */
  math?: string;
  /** ~12 lines of PyTorch-ish pseudocode. */
  code?: { lang: 'python'; src: string };
  /** Tensor ids from the model's tensor catalog (changed from the brief's inline TensorSpec[]). */
  produces?: readonly string[];
  consumes?: readonly string[];
  cost?: (m: ModelSpec, s: Scenario) => NodeCost;
  metrics?: (m: ModelSpec, s: Scenario) => readonly Metric[];
}

/**
 * Edges are authored at the finest depth they make sense. When an endpoint is
 * hidden at the current depth the renderer lifts it to its nearest visible
 * ancestor and drops duplicates and self-loops, so one authored edge serves
 * every depth.
 */
export interface FlowEdge {
  from: string;
  to: string;
  label?: string;
  kind: 'data' | 'control' | 'collective';
  /** Tensor travelling along this edge (hover highlights it; its bytes set the stroke weight). */
  tensorId?: string;
  minDepth: Depth;
}

export interface LensGraph {
  intro: string;
  nodes: readonly FlowNode[];
  edges: readonly FlowEdge[];
}

/**
 * A shared, model-agnostic lens template. Lives in src/lenses/, reads
 * `m.config`, and must not mention any model id.
 */
export type LensTemplate = (m: ModelSpec) => LensGraph;

/**
 * Per-model override hook for structure that is genuinely unique to one model
 * and not expressible as a config feature (e.g. DeepSeek-V3's multi-token
 * prediction head). Applied after the template, in this order:
 * remove → patch → add.
 */
export interface LensPatch {
  /** Appended to the template's intro. */
  intro?: string;
  /** Incident edges are removed too. */
  removeNodes?: readonly string[];
  patchNodes?: Readonly<Record<string, Partial<Omit<FlowNode, 'id'>>>>;
  addNodes?: readonly FlowNode[];
  addEdges?: readonly FlowEdge[];
}

export interface ModelOverrides {
  /** Extra tensors, or replacements for catalog tensors with the same id. */
  tensors?: readonly TensorSpec[];
  lenses?: Partial<Record<LensId, LensPatch>>;
}

// ──────────────────────────────── Model ────────────────────────────────

export interface SourceRef {
  /** Dotted path into the ModelSpec this source backs, e.g. 'config.attention.kvHeads' or 'training.tokens'. A prefix covers everything under it. */
  field: string;
  url?: string;
  /** Not confidently verified: the UI shows an "approximate" marker on this value and anything derived from it. */
  approx?: boolean;
  note?: string;
}

export interface TokenizerSpec {
  algorithm: string;
  /** Tokenizer vocabulary, including special tokens. */
  vocab: number;
  note?: string;
}

export interface TrainingFacts {
  /** Pretraining tokens. */
  tokens: number;
  gpuHours?: number;
  gpu?: string;
}

/**
 * Changed from the brief: no `lenses` field. Lens graphs are generated from
 * shared templates (`LensTemplate`) so four models do not mean four hand-authored
 * copies; `overrides` is the per-model hook for genuinely different structure.
 * Adding a model = adding one file under src/data/models/ that default-exports this.
 */
export interface ModelSpec {
  /** URL slug; must equal the file name under src/data/models/. */
  id: string;
  label: string;
  family: string;
  org: string;
  /** YYYY-MM. */
  released: string;
  /** One paragraph: what makes this model worth studying next to the others. */
  blurb: string;
  config: ModelConfig;
  tokenizer: TokenizerSpec;
  training?: TrainingFacts;
  sources: readonly SourceRef[];
  overrides?: ModelOverrides;
}

// ─────────────────────────────── Hardware ───────────────────────────────

export interface GpuSpec {
  id: string;
  label: string;
  arch: string;
  memory: {
    kind: string;
    bytes: number;
    /** bytes/s */
    bandwidth: number;
  };
  /** Dense (no 2:4 sparsity) peak tensor-core throughput by operand dtype, FLOP/s (OP/s for ints). Absent = unsupported. */
  peakFlops: Readonly<Partial<Record<Dtype, number>>>;
  sms: number;
  /** Combined L1 / shared memory per SM, bytes. */
  sramPerSm: number;
  l2Bytes: number;
  /** bytes/s per direction. */
  links: { nvlink: number | null; pcie: number };
  sources: readonly SourceRef[];
}

// ─────────────────────────────── Memory lens ───────────────────────────────

export type BudgetBucket = MemoryBucket | 'runtime';

/** Per-GPU HBM budget for the worst-case rank under the current scenario. */
export interface HbmBudget {
  buckets: Readonly<Record<BudgetBucket, Derived>>;
  used: Derived;
  capacity: Derived;
  fits: boolean;
}
