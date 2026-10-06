// All numbers on the page are computed here from models.js, hardware.js and the scenario S.
// Scenario S: { phase, gpu, dp, tp, pp, ep, cp, zero, recompute, seq, mbs, gbs, batch, prompt, output, wdtype }
//
// Conventions
// - Rank order (innermost first): TP, CP, DP, PP. A group whose ranks fit in one node uses NVLink, otherwise the NIC.
// - Ring algorithms: all-reduce sends 2(n−1)/n · M per GPU; all-gather, reduce-scatter and all-to-all send (n−1)/n · M.
// - Training precision (Megatron-style BF16 mixed precision): BF16 weights, FP32 gradient buffer,
//   FP32 master weights, FP32 AdamW moments, unless the model sets trainPrecision.
// - Experts: split over EP ranks when ep > 1, otherwise TP-split like dense weights.
// - Estimates are marked `approx` where the formula is a model of framework behaviour rather than exact accounting.

const BYTES = { bf16: 2, fp8: 1, fp32: 4 };
const RUNTIME_RESERVE = 1 * 2 ** 30;   // CUDA context, NCCL / cuBLAS workspaces (approximate)

const ceil = Math.ceil;
const kvHeads = (m) => (m.attention.kind === 'MHA' ? m.attention.heads : m.attention.kvHeads);

function attnDims(m) {
  const a = m.attention;
  return a.kind === 'MLA'
    ? { heads: a.heads, kv: null, dqk: a.nopeDim + a.ropeDim, dv: a.vDim }
    : { heads: a.heads, kv: kvHeads(m), dqk: a.headDim, dv: a.headDim };
}

// ---------------- parameters ----------------
function params(m) {
  const H = m.hidden, L = m.layers, a = m.attention, f = m.ffn;
  let attn;
  if (a.kind === 'MLA') {
    const dqk = a.nopeDim + a.ropeDim;
    attn = H * a.qRank + a.qRank + a.qRank * a.heads * dqk
      + H * (a.kvRank + a.ropeDim) + a.kvRank + a.kvRank * a.heads * (a.nopeDim + a.vDim)
      + a.heads * a.vDim * H;
  } else {
    const q = a.heads * a.headDim, k = kvHeads(m) * a.headDim;
    attn = 2 * H * q + 2 * H * k + (a.bias === 'qkv' ? q + 2 * k : a.bias === 'all' ? q + 2 * k + H : 0);
  }
  const mlp = (F, gated, bias) => (gated ? 3 * H * F + (bias ? 2 * F + H : 0) : 2 * H * F + (bias ? F + H : 0));
  const normP = m.norm === 'LayerNorm' ? 2 * H : H;
  let denseLayers, moeLayers, ffnDenseLayer = 0, perExpert = 0, router = 0, ffnTotal, ffnActive;
  if (f.kind === 'dense') {
    denseLayers = L; moeLayers = 0;
    ffnDenseLayer = mlp(f.inter, f.gated, f.bias);
    ffnTotal = ffnActive = L * ffnDenseLayer;
  } else {
    denseLayers = f.denseLayers; moeLayers = L - f.denseLayers;
    ffnDenseLayer = f.denseLayers ? mlp(f.denseInter, f.gated, false) : 0;
    perExpert = mlp(f.expertInter, f.gated, false);
    router = f.experts * H + (f.routerBias ? f.experts : 0);
    ffnTotal = denseLayers * ffnDenseLayer + moeLayers * (router + (f.experts + f.shared) * perExpert);
    ffnActive = denseLayers * ffnDenseLayer + moeLayers * (router + (f.topK + f.shared) * perExpert);
  }
  const embed = m.vocab * H;
  const pos = m.position === 'learned' ? m.context * H : 0;
  const head = m.tied ? 0 : m.vocab * H;
  const norms = (2 * L + 1) * normP;
  const total = embed + pos + head + L * attn + ffnTotal + norms;
  const routed = f.kind === 'moe' ? moeLayers * f.experts * perExpert : 0;
  const active = total - ffnTotal + ffnActive;
  return {
    embed, pos, head, norms, attnLayer: attn, ffnDenseLayer, perExpert, router, ffnTotal,
    denseLayers, moeLayers, total, active, routed, dense: total - routed,
    matmul: active - embed - pos + (m.tied ? embed : 0),          // the N in 2·N·T and 6·N·T
    layerAvg: (L * attn + ffnTotal + 2 * L * normP) / L,
  };
}

// Values cached per token per layer (K and V, or the MLA latent).
function kvPerTokenLayer(m) {
  const a = m.attention;
  return a.kind === 'MLA' ? a.kvRank + a.ropeDim : 2 * kvHeads(m) * a.headDim;
}

// Activation elements saved for backward per token per layer (FlashAttention: no T×T scores stored).
function savedElemsPerTokenLayer(m, recompute) {
  const H = m.hidden, f = m.ffn, d = attnDims(m), a = m.attention;
  if (recompute === 'full') return H;
  const attn = 2 * H + (a.kind === 'MLA'
    ? a.qRank + 2 * d.heads * d.dqk + (a.kvRank + a.ropeDim) + 2 * d.heads * d.dv
    : 2 * d.heads * d.dqk + 2 * d.kv * d.dqk);
  const per = (F) => (f.gated ? 4 : 2) * F;
  if (f.kind === 'dense') return attn + 2 * H + per(f.inter);
  const P = params(m);
  const moe = 2 * H + f.experts + (f.topK + f.shared) * per(f.expertInter) + 2 * f.topK * H;
  const dense = f.denseLayers ? 2 * H + per(f.denseInter) : 0;
  return attn + (P.denseLayers * dense + P.moeLayers * moe) / m.layers;
}

// Peak live activation elements per token while one layer runs (inference).
function workElemsPerToken(m) {
  const H = m.hidden, f = m.ffn, d = attnDims(m);
  const attn = 2 * d.heads * d.dqk + 2 * (d.kv ?? d.heads) * d.dqk;
  const ffn = f.kind === 'dense' ? (f.gated ? 3 : 2) * f.inter
    : f.experts + (f.topK + f.shared) * (f.gated ? 3 : 2) * f.expertInter + 2 * f.topK * H;
  return 2 * H + Math.max(attn, ffn);
}

function expertsTouched(m, tokens) {
  const f = m.ffn;
  return f.kind === 'moe' ? f.experts * (1 - Math.pow(1 - f.topK / f.experts, tokens)) : 0;
}

// ---------------- layout ----------------
const gpuOf = (S) => window.GPUS.find((g) => g.id === S.gpu) || window.GPUS[0];
const nGPUs = (S) => S.dp * S.tp * S.pp * S.cp;

// Inside a node, collectives can use all of a GPU's NVLinks; a P2P send between two GPUs uses only that pair's links.
function link(S, dim, p2p = false) {
  const G = gpuOf(S), node = G.perNode;
  // EP ranks are taken across DP×CP (stride TP) when they fit there, otherwise across TP×DP×CP contiguously.
  const span = { tp: S.tp, cp: S.tp * S.cp, dp: S.tp * S.cp * S.dp, ep: S.ep <= S.dp * S.cp ? S.tp * S.ep : S.ep, pp: nGPUs(S) }[dim];
  if (span > node) return { name: G.nicName, bw: G.nic, intra: false };
  return p2p ? { name: `${G.nvlinkName}, one pair`, bw: G.nvlinkPair, intra: true } : { name: G.nvlinkName, bw: G.nvlink, intra: true };
}

function layout(m, S) {
  const P = params(m);
  const g = m.ffn.kind === 'moe' && S.ep > 1 ? S.ep : S.tp;     // ranks one layer's experts are split over
  const micro = S.phase === 'training' ? S.gbs / (S.dp * S.mbs) : 1;
  return {
    N: nGPUs(S), nodes: ceil(nGPUs(S) / gpuOf(S).perNode), g,
    denseShard: P.dense / (S.tp * S.pp),                // params per GPU outside routed experts
    expertShard: P.routed / (S.pp * g),                 // routed-expert params per GPU
    expertsPerGPU: m.ffn.kind === 'moe' ? m.ffn.experts / (S.ep > 1 ? S.ep : 1) : 0,
    dpDense: S.dp * S.cp,                                // ranks holding identical dense shards
    dpExpert: (S.dp * S.cp * S.tp) / g,                 // ranks holding identical expert shards
    layersPerStage: ceil(m.layers / S.pp),
    moeLayersPerStage: ceil(P.moeLayers / S.pp),
    micro, inflight: Math.min(S.pp, micro),
    bReplica: ceil(S.batch / S.dp),
  };
}

function validate(m, S) {
  const out = [], d = attnDims(m);
  if (d.heads % S.tp) out.push(`TP ${S.tp} does not divide ${d.heads} query heads.`);
  if (d.kv !== null && S.tp > d.kv) out.push(`TP ${S.tp} > ${d.kv} KV heads: each KV head is replicated on ${S.tp / d.kv} ranks.`);
  if (S.tp > gpuOf(S).perNode) out.push(`TP ${S.tp} spans ${ceil(S.tp / gpuOf(S).perNode)} nodes of ${gpuOf(S).perNode}: TP collectives run over the NIC instead of NVLink.`);
  if (S.pp > m.layers) out.push(`PP ${S.pp} exceeds ${m.layers} layers.`);
  if (m.ffn.kind === 'moe') {
    if (m.ffn.experts % S.ep) out.push(`EP ${S.ep} does not divide ${m.ffn.experts} experts.`);
    if ((S.dp * S.cp * S.tp) % S.ep) out.push(`EP ${S.ep} must divide DP·CP·TP = ${S.dp * S.cp * S.tp}.`);
  } else if (S.ep > 1) out.push('EP applies only to MoE models; set EP = 1.');
  const seqLen = S.phase === 'training' ? S.seq : S.prompt;
  if (seqLen % S.cp) out.push(`CP ${S.cp} does not divide the sequence length ${seqLen}.`);
  if (S.phase === 'training' && S.gbs % (S.dp * S.mbs)) out.push(`Global batch ${S.gbs} is not divisible by DP × micro-batch = ${S.dp * S.mbs}.`);
  if (S.phase === 'inference' && S.wdtype === 'fp8' && !gpuOf(S).flops.fp8) out.push(`${gpuOf(S).name} has no FP8 tensor cores: FP8 weights are dequantized and computed in BF16.`);
  if (S.phase === 'inference' && S.prompt + S.output > m.context) out.push(`Prompt + output exceeds the ${m.context}-token context.`);
  const sys = window.SYSTEM;
  if (sys && nGPUs(S) > sys.gpus) out.push(`Layout needs ${nGPUs(S).toLocaleString('en-US')} GPUs; ${sys.name} has ${sys.gpus} (${sys.gpuNodes} nodes, snapshot ${sys.asOf}).`);
  return out;
}

// ---------------- memory: training ----------------
function trainPrecision(m) {
  const mom = m.trainPrecision?.moments === 'bf16' ? 2 : 4;
  return { w: 2, grad: 4, master: 4, moments: 2 * mom, momentDtype: mom === 2 ? 'BF16' : 'FP32' };
}

function trainingMemory(m, S) {
  const P = params(m), Ly = layout(m, S), tp = trainPrecision(m), z = S.zero, moe = m.ffn.kind === 'moe';
  const shard = (zMin) => Ly.denseShard / (z >= zMin ? Ly.dpDense : 1) + Ly.expertShard / (z >= zMin ? Ly.dpExpert : 1);
  const by = (zMin) => `TP×PP${moe ? (S.ep > 1 ? ', experts EP×PP' : ', experts TP×PP') : ''}${z >= zMin && Ly.dpDense > 1 ? `, DP (ZeRO-${zMin === 3 ? 3 : z})` : ''}`;
  const tokensMicro = S.mbs * S.seq / S.cp;
  const persistent = [
    { name: 'Weights', dtype: 'BF16', shardedBy: by(3), bytes: shard(3) * tp.w, lifetime: 'allocated at load; read every forward and backward', tech: 'mixed-precision' },
    { name: 'Master weights (optimizer)', dtype: 'FP32', shardedBy: by(1), bytes: shard(1) * tp.master, lifetime: 'allocated at init; updated once per step', tech: 'zero' },
    { name: 'AdamW moments m, v', dtype: tp.momentDtype, shardedBy: by(1), bytes: shard(1) * tp.moments, lifetime: 'allocated at init; updated once per step', tech: 'zero' },
  ];
  const perStep = [
    { name: 'Gradients', dtype: 'FP32', shardedBy: by(2), bytes: shard(2) * tp.grad, lifetime: 'accumulated during backward over all micro-batches; consumed by the optimizer step', tech: 'zero' },
    { name: 'Saved activations', dtype: 'BF16', shardedBy: `TP (with SP)${S.cp > 1 ? ', CP' : ''}, PP`,
      bytes: Ly.layersPerStage * Ly.inflight * tokensMicro * savedElemsPerTokenLayer(m, S.recompute) * 2 / S.tp,
      lifetime: `written in forward, read in backward; ${Ly.inflight} micro-batch${Ly.inflight > 1 ? 'es' : ''} in flight on the first PP stage`, approx: true, tech: 'activation-checkpointing',
      detail: `${Ly.layersPerStage} layers × ${Ly.inflight} in flight × ${fmtN(tokensMicro)} tokens × ${fmtN(savedElemsPerTokenLayer(m, S.recompute))} values/token/layer × 2 B / TP ${S.tp}` },
    { name: 'Logits for the loss', dtype: 'FP32', shardedBy: 'TP (vocab-parallel)', bytes: tokensMicro * m.vocab * 4 / S.tp,
      lifetime: 'one micro-batch, last PP stage', approx: true },
  ];
  if (z >= 3) perStep.push({ name: 'Gathered layer weights (ZeRO-3)', dtype: 'BF16', shardedBy: 'TP', bytes: 2 * P.layerAvg / S.tp * 2,
    lifetime: 'one layer (+1 prefetched) all-gathered before use, freed after', approx: true, tech: 'fsdp' });
  perStep.push({ name: 'Runtime reserve', dtype: '—', shardedBy: '—', bytes: RUNTIME_RESERVE, lifetime: 'CUDA context, NCCL / cuBLAS workspaces', approx: true });
  return finishMemory(S, persistent, perStep);
}

// ---------------- memory: inference ----------------
function kvBytesPerTokenGPU(m, S) {
  const a = m.attention, Ly = layout(m, S);
  const perLayer = a.kind === 'MLA' ? a.kvRank + a.ropeDim : 2 * ceil(kvHeads(m) / S.tp) * a.headDim;   // MLA latent is replicated over TP
  return Ly.layersPerStage * perLayer * BYTES.bf16 / S.cp;
}

function inferenceMemory(m, S) {
  const Ly = layout(m, S), wb = BYTES[S.wdtype], ctx = S.prompt + S.output, moe = m.ffn.kind === 'moe';
  const persistent = [
    { name: 'Weights', dtype: S.wdtype.toUpperCase(), shardedBy: `TP×PP${moe ? (S.ep > 1 ? ', experts EP×PP' : ', experts TP×PP') : ''}; replicated across DP`,
      bytes: (Ly.denseShard + Ly.expertShard) * wb, lifetime: 'allocated at load; read every forward pass' },
  ];
  const perStep = [
    { name: 'KV cache', dtype: 'BF16', shardedBy: m.attention.kind === 'MLA' ? 'PP (latent replicated across TP)' : 'TP (by KV head), PP', bytes: kvBytesPerTokenGPU(m, S) * Ly.bReplica * ctx,
      lifetime: `per request; grows by one position per generated token (${Ly.bReplica} sequences × ${fmtN(ctx)} tokens)`, tech: 'kv-cache',
      detail: `${fmtBytes(kvBytesPerTokenGPU(m, S))} per token per GPU` },
    { name: 'Prefill activation workspace', dtype: 'BF16', shardedBy: 'TP', bytes: Ly.bReplica * (S.prompt / S.cp) * workElemsPerToken(m) * 2 / S.tp,
      lifetime: 'one layer at a time during prefill; chunked prefill caps it', approx: true },
    { name: 'Next-token logits', dtype: 'FP32', shardedBy: 'TP (vocab-parallel)', bytes: Ly.bReplica * m.vocab * 4 / S.tp, lifetime: 'per decode step' },
    { name: 'Runtime reserve', dtype: '—', shardedBy: '—', bytes: RUNTIME_RESERVE, lifetime: 'CUDA context, NCCL / cuBLAS workspaces', approx: true },
  ];
  return finishMemory(S, persistent, perStep);
}

function finishMemory(S, persistent, perStep) {
  const sum = (rows) => rows.reduce((s, r) => s + r.bytes, 0);
  const total = sum(persistent) + sum(perStep);
  const cap = gpuOf(S).hbm;
  return { persistent, perStep, persistentTotal: sum(persistent), perStepTotal: sum(perStep), total, cap, fits: total <= cap };
}

// KV cache + weights per GPU as the sequence grows, for the KV growth chart.
function kvGrowth(m, S) {
  const mem = inferenceMemory(m, S), Ly = layout(m, S);
  const fixed = mem.total - mem.perStep[0].bytes;           // everything except the KV cache
  const perTok = kvBytesPerTokenGPU(m, S) * Ly.bReplica;     // bytes per position, all sequences on this replica
  const maxCtx = Math.max(0, Math.floor((mem.cap - fixed) / perTok));
  return { fixed, perTok, maxCtx, cap: mem.cap, current: S.prompt + S.output };
}

// ---------------- compute: inference steps ----------------
function computeDtype(S) { return S.wdtype === 'fp8' && gpuOf(S).flops.fp8 ? 'fp8' : 'bf16'; }

function weightBytesRead(m, S, tokens) {
  const P = params(m), wb = BYTES[S.wdtype];
  return m.ffn.kind === 'moe' ? (P.dense + P.moeLayers * expertsTouched(m, tokens) * P.perExpert) * wb : P.total * wb;
}

function inferenceSteps(m, S) {
  const P = params(m), d = attnDims(m), Ly = layout(m, S), G = gpuOf(S);
  const b = Ly.bReplica, Sp = S.prompt, ctx = S.prompt + S.output, L = m.layers;
  const kvTok = kvPerTokenLayer(m) * L * BYTES.bf16;                // KV bytes per token, whole model
  const pi = G.flops[computeDtype(S)], beta = G.hbmBW;
  const prof = (F, Q, nPar, tokensOut) => {
    const tc = F / (nPar * pi), tm = Q / (nPar * beta), t = Math.max(tc, tm);
    return { F, Q, I: F / Q, tc, tm, t, bound: tc >= tm ? 'compute-bound' : 'memory-bandwidth-bound', tps: tokensOut / t };
  };
  // Prefill: all prompt positions at once; causal attention counts half of the T×T score matrix.
  const Fp = 2 * P.matmul * b * Sp + L * b * d.heads * Sp * Sp * (d.dqk + d.dv);
  const Qp = weightBytesRead(m, S, b * Sp) + b * Sp * kvTok;
  // Decode: one position per sequence, attending over the full context (last step).
  const Fd = 2 * P.matmul * b + 2 * L * b * d.heads * ctx * (d.dqk + d.dv);
  const Qd = weightBytesRead(m, S, b) + b * ctx * kvTok;
  return {
    ridge: pi / beta, pi, beta, dtype: computeDtype(S),
    prefill: prof(Fp, Qp, S.tp * S.cp, b * Sp),
    decode: prof(Fd, Qd, S.tp, b),
  };
}

function trainingCompute(m, S) {
  const P = params(m), d = attnDims(m), G = gpuOf(S);
  const c = S.recompute === 'full' ? 8 : 6, tokens = S.gbs * S.seq;
  const attnFwd = m.layers * S.gbs * d.heads * S.seq * S.seq * (d.dqk + d.dv);   // causal: half of 2·…
  const F = c * P.matmul * tokens + (c / 2) * attnFwd;
  const perGPU = F / nGPUs(S);
  return { F, perGPU, tMin: perGPU / G.flops.bf16, c, tokens };
}

// ---------------- collectives ----------------
const AR = (n, M) => (2 * (n - 1) / n) * M;
const AG = (n, M) => ((n - 1) / n) * M;

function row(S, dim, op, when, msg, count, perOp, tech) {
  const l = link(S, dim, op.startsWith('P2P'));
  const volume = count * perOp;
  return { dim, op, when, msg, count, volume, link: l.name, intra: l.intra, time: volume / l.bw, tech };
}

function trainingCollectives(m, S) {
  const Ly = layout(m, S), H = m.hidden, out = [], tp = trainPrecision(m), m_ = Ly.micro;
  const tokens = S.mbs * S.seq / S.cp, act = tokens * H * 2;
  if (S.tp > 1) out.push(row(S, 'tp', 'all-reduce (reduce-scatter + all-gather with SP)', 'after attention W_O and after FFN W_down; forward and backward',
    act, 4 * Ly.layersPerStage * m_, AR(S.tp, act), 'tensor-parallel'));
  if (S.cp > 1) {
    const chunk = tokens * (m.attention.kind === 'MLA' ? kvPerTokenLayer(m) : kvPerTokenLayer(m) / Math.min(S.tp, kvHeads(m))) * 2;
    out.push(row(S, 'cp', 'P2P ring (send/recv K, V blocks)', 'inside attention: K, V forward; K, V and dK, dV backward',
      chunk, 3 * (S.cp - 1) * Ly.layersPerStage * m_, chunk, 'context-parallel'));
  }
  if (S.pp > 1) out.push(row(S, 'pp', 'P2P send/recv', 'stage boundary: activations forward, activation gradients backward',
    act / S.tp, 2 * m_, act / S.tp, 'pipeline-parallel'));
  if (m.ffn.kind === 'moe' && S.ep > 1) {
    const M = (tokens / S.tp) * m.ffn.topK * H * 2;
    out.push(row(S, 'ep', 'all-to-all', 'MoE layers: dispatch tokens to experts and combine outputs; forward and backward',
      M, 4 * Ly.moeLayersPerStage * m_, AG(S.ep, M), 'expert-parallel'));
  }
  const dpRows = (n, shard, label) => {
    if (n <= 1 || shard === 0) return;
    const G = shard * tp.grad, W = shard * tp.w;
    if (S.zero === 0) out.push(row(S, 'dp', 'all-reduce', `${label}gradient sync after the last micro-batch`, G, 1, AR(n, G), 'data-parallel'));
    else if (S.zero < 3) {
      out.push(row(S, 'dp', 'reduce-scatter', `${label}gradients → each rank keeps 1/${n}`, G, 1, AG(n, G), 'zero'));
      out.push(row(S, 'dp', 'all-gather', `${label}updated BF16 weights after the optimizer step`, W, 1, AG(n, W), 'zero'));
    } else {
      out.push(row(S, 'dp', 'all-gather', `${label}weights before each layer, forward and backward, every micro-batch`, W, 2 * m_, AG(n, W), 'fsdp'));
      out.push(row(S, 'dp', 'reduce-scatter', `${label}gradients`, G, 1, AG(n, G), 'fsdp'));
    }
  };
  dpRows(Ly.dpDense, Ly.denseShard, m.ffn.kind === 'moe' ? 'dense params: ' : '');
  dpRows(Ly.dpExpert, Ly.expertShard, 'expert params: ');
  return out;
}

function inferenceCollectives(m, S, step) {
  const Ly = layout(m, S), H = m.hidden, out = [];
  const tokens = (step === 'prefill' ? Ly.bReplica * S.prompt : Ly.bReplica) / (step === 'prefill' ? S.cp : 1);
  const act = tokens * H * 2;
  if (S.tp > 1) out.push(row(S, 'tp', 'all-reduce', 'after attention W_O and after FFN W_down', act, 2 * Ly.layersPerStage, AR(S.tp, act), 'tensor-parallel'));
  if (m.ffn.kind === 'moe' && S.ep > 1) {
    const M = (tokens / S.tp) * m.ffn.topK * H * 2;
    out.push(row(S, 'ep', 'all-to-all', 'MoE layers: dispatch and combine', M, 2 * Ly.moeLayersPerStage, AG(S.ep, M), 'expert-parallel'));
  }
  if (S.pp > 1) out.push(row(S, 'pp', 'P2P send/recv', 'stage boundary: activations', act, 1, act, 'pipeline-parallel'));
  if (step === 'prefill' && S.cp > 1) {
    const chunk = tokens * (m.attention.kind === 'MLA' ? kvPerTokenLayer(m) : kvPerTokenLayer(m) / Math.min(S.tp, kvHeads(m))) * 2;
    out.push(row(S, 'cp', 'P2P ring (K, V blocks)', 'inside attention during prefill', chunk, (S.cp - 1) * Ly.layersPerStage, chunk, 'context-parallel'));
  }
  return out;
}

// ---------------- transfers (bytes moved, link, time) ----------------
function transfers(m, S) {
  const G = gpuOf(S), Ly = layout(m, S), rows = [];
  // No node-local disk: checkpoints come from network storage over the NIC, then CPU memory → NVLink-C2C → HBM.
  // The slower hop bounds the load; storage-server throughput is not modelled.
  const storage = window.SYSTEM?.storage ?? 'storage';
  const loadPath = `${storage.split(' (')[0]} → NIC → ${G.hostMemName ?? 'CPU memory'} → ${G.hostName} → HBM`;
  const loadLink = `${G.nicName} → ${G.hostName}`, loadBW = Math.min(G.nic, G.host);
  if (S.phase === 'training') {
    const mem = trainingMemory(m, S), tp = trainPrecision(m);
    const wShard = (Ly.denseShard + Ly.expertShard) * tp.w;                                   // resident TP×PP(×EP) shard before ZeRO-3
    const optShard = Ly.denseShard / (S.zero >= 1 ? Ly.dpDense : 1) + Ly.expertShard / (S.zero >= 1 ? Ly.dpExpert : 1);
    const actMicro = mem.perStep[1].bytes / Ly.inflight;
    rows.push(['Load', loadPath, 'checkpoint shard of this rank: BF16 weights + optimizer state', mem.persistentTotal, loadLink, loadBW]);
    rows.push(['Each step', 'HBM → SMs (forward)', `weights streamed once per micro-batch (${fmtN(Ly.micro)} micro-batches)`, Ly.micro * wShard, 'HBM', G.hbmBW]);
    rows.push(['Each step', 'HBM ↔ SMs (backward)', 'weights read + FP32 gradient accumulate (read + write) per micro-batch', Ly.micro * (wShard + 2 * (wShard / tp.w) * tp.grad), 'HBM', G.hbmBW]);
    rows.push(['Each step', 'HBM ↔ SMs (activations)', 'saved activations written in forward and read in backward', Ly.micro * 2 * actMicro, 'HBM', G.hbmBW]);
    rows.push(['Each step', 'HBM ↔ SMs (optimizer)', 'AdamW: read master, m, v, grad; write master, m, v, BF16 weight', optShard * (2 * tp.master + 2 * tp.moments + tp.grad + tp.w), 'HBM', G.hbmBW]);
  } else {
    const mem = inferenceMemory(m, S), st = inferenceSteps(m, S);
    rows.push(['Load', loadPath, 'weights of this rank', mem.persistent[0].bytes, loadLink, loadBW]);
    rows.push(['Prefill', 'HBM → SMs', 'weights read once for all prompt positions + KV cache written', st.prefill.Q / (S.tp * S.pp * S.cp), 'HBM', G.hbmBW]);
    rows.push(['Decode step', 'HBM → SMs', 'weights read again + entire KV cache read, for one new position', st.decode.Q / (S.tp * S.pp), 'HBM', G.hbmBW]);
  }
  return rows.map(([phase, path, what, bytes, linkName, bw]) => ({ phase, path, what, bytes, link: linkName, time: bytes / bw }));
}

// ---------------- formatting (shared with app.js) ----------------
function fmtN(n) { return Math.round(n).toLocaleString('en-US'); }
function fmtBytes(n) {
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2)} ${u[i]}`;
}
function fmtParams(n) { return n >= 1e9 ? `${(n / 1e9).toFixed(n >= 100e9 ? 0 : 2)}B` : `${(n / 1e6).toFixed(n >= 100e6 ? 0 : 1)}M`; }
function fmtFlops(n) {
  const u = [[1e18, 'EFLOP'], [1e15, 'PFLOP'], [1e12, 'TFLOP'], [1e9, 'GFLOP']];
  for (const [s, name] of u) if (n >= s) return `${(n / s).toFixed(n / s >= 100 ? 0 : 1)} ${name}`;
  return `${fmtN(n)} FLOP`;
}
function fmtBW(bytesPerSec) { return bytesPerSec >= 1e12 ? `${(bytesPerSec / 1e12).toFixed(2)} TB/s` : `${(bytesPerSec / 1e9).toFixed(0)} GB/s`; }
function fmtTime(s) { return s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(s >= 0.1 ? 0 : 2)} ms` : `${(s * 1e6).toFixed(0)} µs`; }
