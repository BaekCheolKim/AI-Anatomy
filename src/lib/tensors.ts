/**
 * The tensor catalog: every tensor a lens can show, defined once per model and
 * generated from config features. Lenses reference tensors by id, so the KV
 * cache on the Inference graph and the KV bar in the Memory lens are the same
 * object with the same bytes.
 */
import type { Derived, ModelSpec, Scenario, Shard, TensorSpec, Workload } from '../types';
import { bytesPerParam, denseMlp, derive, kvBytes, resolveDtype, weightBytes } from './calc';
import { dimEnv } from './dims';
import { evalExpr } from './expr';

const ACT = { role: 'act' } as const;
const KV = { role: 'kv' } as const;
const WT = { role: 'weight' } as const;

function build(m: ModelSpec): TensorSpec[] {
  const cfg = m.config;
  const a = cfg.attention;
  const isMla = a.kind === 'MLA';
  const isMoe = cfg.ffn.kind === 'moe';
  const hd = isMla ? 'Dqk' : 'D';
  const vd = isMla ? 'Dv' : 'D';
  const kvh = isMla ? 'Nh' : 'Nkv';
  const dense = denseMlp(cfg);
  const t: TensorSpec[] = [];

  // ── input ──
  t.push({
    id: 'token-ids',
    label: 'Token IDs',
    dtype: 'int32',
    shape: ['B', 'S'],
    location: 'CPU RAM',
    sharding: [{ axis: 'dp', dim: 'B' }],
    lifetime: 'per-request',
    movement: 'Written by the tokenizer on the CPU and copied host→device over PCIe once per request. A few KiB, never the bottleneck.',
  });
  t.push({
    id: 'embed-table',
    label: cfg.tiedEmbeddings ? 'Token embedding (tied with LM head)' : 'Token embedding',
    dtype: WT,
    shape: ['V', 'H'],
    location: 'GPU HBM',
    sharding: [{ axis: 'tp', dim: 'V' }],
    lifetime: 'persistent',
    movement: 'Gathered, not multiplied: each token id selects one H-wide row. Under TP each rank owns a slice of the vocabulary.',
    note: cfg.tiedEmbeddings ? 'The same matrix is reused, transposed, as the LM head.' : undefined,
  });
  if (cfg.position.kind === 'learned') {
    t.push({
      id: 'pos-table',
      label: 'Learned position embedding',
      dtype: WT,
      shape: ['Pos', 'H'],
      location: 'GPU HBM',
      lifetime: 'persistent',
      movement: 'Row p is added to the token embedding at position p. Fixed size: sequences longer than Pos cannot be represented.',
    });
  }
  t.push({
    id: 'hidden',
    label: 'Hidden state (residual stream)',
    dtype: ACT,
    shape: ['B', 'S', 'H'],
    location: 'GPU HBM',
    sharding: [{ axis: 'dp', dim: 'B' }],
    ownerNote: 'Full width on every TP rank. Sequence parallelism would split it along S between the TP all-reduces.',
    lifetime: 'transient',
    movement: 'Flows block to block. Each sub-block reads it, and its output is added back (the residual) before the next block.',
  });

  // ── attention ──
  if (a.kind === 'MLA') {
    if (a.qLoraRank !== null) {
      t.push({
        id: 'q-latent',
        label: 'Query latent c_Q',
        dtype: ACT,
        shape: ['B', 'S', 'Rq'],
        location: 'GPU HBM',
        sharding: [{ axis: 'dp', dim: 'B' }],
        lifetime: 'transient',
        movement: 'Down-projected from the hidden state, normalised, then up-projected to per-head queries.',
      });
    }
    t.push({
      id: 'kv-latent',
      label: 'KV latent c_KV ‖ k_rope',
      dtype: ACT,
      shape: ['B', 'S', 'Rkv+Dr'],
      location: 'GPU HBM',
      sharding: [{ axis: 'dp', dim: 'B' }],
      lifetime: 'transient',
      movement: 'One compressed vector per token shared by all heads, plus a small RoPE key. This is what gets written to the KV cache.',
    });
  }
  t.push({
    id: 'q',
    label: 'Queries Q',
    dtype: ACT,
    shape: ['B', 'S', 'Nh', hd],
    location: 'GPU HBM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: 'Nh' },
    ],
    lifetime: 'transient',
    movement: 'Produced by the Q projection, rotated by RoPE where used, then streamed into the attention kernel tile by tile.',
  });
  t.push({
    id: 'k',
    label: 'Keys K',
    dtype: ACT,
    shape: ['B', 'S', kvh, hd],
    location: 'GPU HBM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: kvh },
    ],
    lifetime: 'transient',
    movement: isMla
      ? 'Reconstructed per head from the cached latent. At decode the up-projection can be absorbed into the query, so K is never materialised.'
      : 'Written into the KV cache at its position, then read back for every later token.',
  });
  t.push({
    id: 'v',
    label: 'Values V',
    dtype: ACT,
    shape: ['B', 'S', kvh, vd],
    location: 'GPU HBM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: kvh },
    ],
    lifetime: 'transient',
    movement: isMla
      ? 'Reconstructed per head from the cached latent. At decode the up-projection folds into the output projection.'
      : 'Written into the KV cache alongside K.',
  });
  t.push({
    id: 'attn-scores',
    label: 'Attention scores softmax(QKᵀ/√d)',
    dtype: 'fp32',
    shape: ['B', 'Nh', 'S', 'S'],
    location: 'GPU SRAM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: 'Nh' },
    ],
    lifetime: 'transient',
    movement: 'FlashAttention computes it tile by tile in on-chip SRAM with an online softmax and never writes it to HBM.',
    note: 'Shown at full logical size so the S² term is visible: this is what naive attention would have to store.',
  });
  t.push({
    id: 'attn-out',
    label: 'Attention output',
    dtype: ACT,
    shape: ['B', 'S', 'Nh', vd],
    location: 'GPU HBM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: 'Nh' },
    ],
    lifetime: 'transient',
    movement: 'Heads are concatenated and fed to the output projection. Under TP each rank holds its heads, and the output projection ends in an all-reduce.',
  });
  t.push({
    id: 'kv-cache',
    label: isMla ? 'KV cache (compressed latent)' : 'KV cache',
    dtype: KV,
    shape: isMla ? ['L', 'B', 'S+G', 'Rkv+Dr'] : ['2', 'L', 'B', 'Nkv', 'S+G', 'D'],
    location: 'GPU HBM',
    sharding: isMla
      ? [
          { axis: 'pp', dim: 'L' },
          { axis: 'dp', dim: 'B' },
        ]
      : [
          { axis: 'pp', dim: 'L' },
          { axis: 'dp', dim: 'B' },
          { axis: 'tp', dim: 'Nkv' },
        ],
    ownerNote: isMla
      ? 'The latent is shared by all heads, so TP cannot split it: every TP rank holds a full copy. This is why MLA models are often served with data-parallel attention.'
      : 'Split by KV head across TP. When tp > Nkv, heads are replicated: each rank still holds one whole head.',
    lifetime: 'per-request',
    movement:
      'Prefill writes all S positions at once. Each decode step appends one position and reads the entire cache back, which is why decode is bandwidth-bound.',
    bucket: 'kv-cache',
    workloads: ['inference'],
    explain: (mm, s) => kvBytes(mm, s),
  });

  // ── FFN ──
  if (dense) {
    const ffnShard: Shard[] = [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: 'F' },
    ];
    if (dense.gated) {
      t.push({
        id: 'mlp-gate',
        label: 'Gate projection',
        dtype: ACT,
        shape: ['B', 'S', 'F'],
        location: 'GPU HBM',
        sharding: ffnShard,
        lifetime: 'transient',
        movement: 'Column-parallel under TP: each rank computes F/tp of it with no communication.',
      });
    }
    t.push({
      id: 'mlp-up',
      label: dense.gated ? 'Up projection' : 'FFN hidden (fc1 output)',
      dtype: ACT,
      shape: ['B', 'S', 'F'],
      location: 'GPU HBM',
      sharding: ffnShard,
      lifetime: 'transient',
      movement: 'Column-parallel under TP: each rank computes F/tp of it with no communication.',
    });
    t.push({
      id: 'mlp-act',
      label: dense.gated ? 'act(gate) ⊙ up' : 'act(fc1)',
      dtype: ACT,
      shape: ['B', 'S', 'F'],
      location: 'GPU HBM',
      sharding: ffnShard,
      lifetime: 'transient',
      movement: 'Elementwise, usually fused into the preceding matmul epilogue. Feeds the row-parallel down projection, which ends in an all-reduce under TP.',
    });
  }
  if (isMoe) {
    t.push({
      id: 'router-logits',
      label: 'Router scores',
      dtype: 'fp32',
      shape: ['B', 'S', 'E'],
      location: 'GPU HBM',
      sharding: [{ axis: 'dp', dim: 'B' }],
      lifetime: 'transient',
      movement: 'One score per expert per token, kept in fp32 so top-k selection is stable.',
    });
    t.push({
      id: 'expert-ids',
      label: 'Selected experts',
      dtype: 'int32',
      shape: ['B', 'S', 'K'],
      location: 'GPU HBM',
      sharding: [{ axis: 'dp', dim: 'B' }],
      lifetime: 'transient',
      movement: 'Drives the all-to-all dispatch: each token is sent to the ranks that own its experts.',
    });
    t.push({
      id: 'expert-in',
      label: 'Dispatched tokens',
      dtype: ACT,
      shape: ['B', 'S', 'K', 'H'],
      location: 'GPU HBM',
      sharding: [{ axis: 'dp', dim: 'B' }],
      ownerNote: 'After the all-to-all, each EP rank holds only the copies routed to its own experts (≈ 1/ep of this when routing is balanced).',
      lifetime: 'transient',
      movement: 'All-to-all over the EP group, then grouped GEMMs per expert.',
    });
    t.push({
      id: 'expert-act',
      label: 'Expert intermediate',
      dtype: ACT,
      shape: ['B', 'S', 'K', 'Fe'],
      location: 'GPU HBM',
      sharding: [{ axis: 'dp', dim: 'B' }],
      lifetime: 'transient',
      movement: 'Inside each expert’s SwiGLU. Never leaves the rank that owns the expert.',
    });
    t.push({
      id: 'expert-out',
      label: 'Expert outputs',
      dtype: ACT,
      shape: ['B', 'S', 'K', 'H'],
      location: 'GPU HBM',
      sharding: [{ axis: 'dp', dim: 'B' }],
      lifetime: 'transient',
      movement: 'All-to-all back to the tokens’ home ranks, then weighted by the router scores and summed.',
    });
  }

  // ── output ──
  t.push({
    id: 'logits',
    label: 'Logits',
    dtype: ACT,
    shape: ['B', 'S', 'V'],
    location: 'GPU HBM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: 'V' },
    ],
    lifetime: 'transient',
    movement: 'Training needs every position for the loss. Generation needs only the last one (next-token logits).',
  });
  t.push({
    id: 'next-logits',
    label: 'Next-token logits',
    dtype: 'fp32',
    shape: ['B', 'V'],
    location: 'GPU HBM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: 'V' },
    ],
    lifetime: 'per-token',
    movement: 'Upcast to fp32 for softmax and sampling. Under TP the vocabulary shards are gathered first.',
    bucket: 'activations',
    workloads: ['inference'],
  });
  t.push({
    id: 'next-token',
    label: 'Sampled token',
    dtype: 'int32',
    shape: ['B'],
    location: 'GPU HBM',
    sharding: [{ axis: 'dp', dim: 'B' }],
    lifetime: 'per-token',
    movement: 'Copied to the CPU to detokenize and check stop conditions, and fed back as the next decode step’s input.',
    workloads: ['inference'],
  });
  t.push({
    id: 'act-workspace',
    label: 'Activation working set',
    dtype: ACT,
    shape: ['B', 'S', 'W'],
    location: 'GPU HBM',
    sharding: [
      { axis: 'dp', dim: 'B' },
      { axis: 'tp', dim: 'W' },
    ],
    lifetime: 'transient',
    movement: 'Peak live activations during prefill of one layer, reused layer after layer.',
    bucket: 'activations',
    workloads: ['inference'],
    note: 'Chunked prefill caps B·S at the chunk size, shrinking this.',
  });

  // ── training state ──
  t.push({
    id: 'saved-acts',
    label: 'Saved activations',
    dtype: ACT,
    shape: ['min(pp,mb)', 'L', 'Bmb', 'S', 'A'],
    location: 'GPU HBM',
    sharding: [
      { axis: 'pp', dim: 'L' },
      { axis: 'tp', dim: 'S' },
    ],
    ownerNote:
      'Per DP replica: Bmb is already a per-replica micro-batch. The TP split along S assumes sequence parallelism. Under 1F1B the first pipeline stage holds min(pp, mb) micro-batches in flight.',
    lifetime: 'per-microbatch',
    movement: 'Written during forward, read once by backward, then freed. Full recompute keeps only each block’s input and re-runs the forward during backward.',
    bucket: 'activations',
    workloads: ['training'],
  });
  t.push({
    id: 'loss-logits',
    label: 'Logits for the loss',
    dtype: 'fp32',
    shape: ['Bmb', 'S', 'V'],
    location: 'GPU HBM',
    sharding: [{ axis: 'tp', dim: 'V' }],
    lifetime: 'per-microbatch',
    movement: 'Vocab-parallel cross-entropy keeps each TP rank on its own vocabulary slice.',
    bucket: 'activations',
    workloads: ['training'],
    note: 'Many frameworks compute the loss in chunks so this is never fully materialised.',
  });

  const paramGroups = isMoe
    ? ([
        { key: '', dim: 'Pd', label: 'non-expert', split: denseSplit('Pd') },
        { key: 'expert-', dim: 'Pe', label: 'routed-expert', split: expertSplit },
      ] as const)
    : ([{ key: '', dim: 'Pd', label: '', split: denseSplit('Pd') }] as const);

  for (const g of paramGroups) {
    const scope = g.label ? ` (${g.label})` : '';
    t.push({
      id: `${g.key}weights`,
      label: `Weights${scope}`,
      dtype: WT,
      shape: [g.dim],
      location: 'GPU HBM',
      sharding: g.split(3),
      lifetime: 'persistent',
      movement:
        'Loaded once (NVMe → CPU RAM → HBM). Every forward step streams them from HBM through the SMs, and at decode that read dominates the step time.',
      bucket: 'weights',
      explain: isMoe ? undefined : (mm, s) => weightBytes(mm, s),
    });
    t.push({
      id: `${g.key}grads`,
      label: `Gradients${scope}`,
      dtype: WT,
      shape: [g.dim],
      location: 'GPU HBM',
      sharding: g.split(2),
      lifetime: 'per-step',
      movement:
        'Produced by backward layer by layer, all-reduced across DP (reduce-scattered under ZeRO-2+), then consumed by the optimizer and freed.',
      bucket: 'gradients',
      workloads: ['training'],
    });
    t.push({
      id: `${g.key}optimizer`,
      label: `AdamW state${scope}`,
      dtype: 'fp32',
      shape: ['Nopt', g.dim],
      location: 'GPU HBM',
      sharding: g.split(1),
      lifetime: 'persistent',
      movement:
        'Read and written once per step, elementwise; never sent anywhere. ZeRO-1+ gives each DP rank one slice to update, then all-gathers the new weights.',
      bucket: 'optimizer',
      workloads: ['training'],
      note: 'Nopt = 3 (m, v, master copy) with 16-bit weights, 2 when training in fp32.',
    });
  }

  const overrides = m.overrides?.tensors ?? [];
  const byId = new Map(t.map((x) => [x.id, x]));
  for (const o of overrides) byId.set(o.id, o);
  return [...byId.values()];
}

function denseSplit(dim: string) {
  return (zeroAtLeast: 1 | 2 | 3): Shard[] => [
    { axis: 'tp', dim },
    { axis: 'pp', dim },
    { axis: 'dp', dim, zeroAtLeast },
  ];
}

function expertSplit(zeroAtLeast: 1 | 2 | 3): Shard[] {
  return [
    { axis: 'pp', dim: 'Pe' },
    { axis: 'ep', dim: 'Pe', ways: 'Eg' },
    { axis: 'dp', dim: 'Pe', ways: 'Edp', zeroAtLeast },
  ];
}

const catalogCache = new WeakMap<ModelSpec, ReadonlyMap<string, TensorSpec>>();

export function tensorCatalog(m: ModelSpec): ReadonlyMap<string, TensorSpec> {
  let c = catalogCache.get(m);
  if (!c) {
    c = new Map(build(m).map((t) => [t.id, t]));
    catalogCache.set(m, c);
  }
  return c;
}

export function tensorExists(t: TensorSpec, w: Workload): boolean {
  return !t.workloads || t.workloads.includes(w);
}

// ─────────────────────────── Sizing ───────────────────────────

export function resolveShape(t: TensorSpec, m: ModelSpec, s: Scenario): number[] {
  const env = dimEnv(m, s);
  return t.shape.map((d) => evalExpr(d, env));
}

/** Logical (unsharded) bytes: numel × bytesPerParam(dtype). */
export function tensorBytes(t: TensorSpec, m: ModelSpec, s: Scenario): Derived {
  const dims = resolveShape(t, m, s);
  const b = bytesPerParam(resolveDtype(t.dtype, s));
  const terms: Record<string, number> = {};
  t.shape.forEach((d, i) => {
    let key = d;
    for (let n = 2; key in terms; n++) key = `${d} (${n})`;
    terms[key] = dims[i] ?? NaN;
  });
  terms.b = b;
  return derive('tensorBytes', dims.reduce((p, x) => p * x, 1) * b, terms);
}

function shardApplies(sh: Shard, s: Scenario): boolean {
  if (sh.zeroAtLeast !== undefined && !(s.workload === 'training' && s.zeroStage >= sh.zeroAtLeast)) return false;
  if (sh.onlyIfOne !== undefined && s[sh.onlyIfOne] !== 1) return false;
  return true;
}

/** Bytes resident on one rank after applying every active shard. */
export function perRankBytes(t: TensorSpec, m: ModelSpec, s: Scenario): Derived {
  const total = tensorBytes(t, m, s);
  const env = dimEnv(m, s);
  const terms: Record<string, number | Derived> = { bytes: total };
  let factor = 1;
  for (const sh of t.sharding ?? []) {
    if (!shardApplies(sh, s)) continue;
    const d = evalExpr(sh.dim, env);
    const w = sh.ways !== undefined ? evalExpr(sh.ways, env) : s[sh.axis];
    if (w <= 1) continue;
    const f = Math.ceil(d / w) / d;
    factor *= f;
    terms[`${sh.dim} ÷ ${sh.ways ?? sh.axis}`] = f;
  }
  return derive('perRankBytes', total.value * factor, terms);
}
