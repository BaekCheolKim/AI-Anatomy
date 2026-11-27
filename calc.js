// Every number on the page is computed here from models.js.
// Results are parameter counts, bytes and FLOPs, so they hold on any hardware. Hardware enters only
// through inputs the reader sets: memory per GPU, number of GPUs and, for placing parallel groups,
// GPUs per node.
//
// Units: bytes are shown in decimal units (1 GB = 10^9 B), so parameters × bytes per parameter reads
// directly in GB. Exact counts appear on hover.

const DTYPE_BYTES = { fp32: 4, bf16: 2, fp16: 2, fp8: 1, mxfp4: 0.53125 };   // MXFP4: 4-bit values + one 8-bit scale per 32 values
const DTYPE_NAME = { fp32: 'FP32', bf16: 'BF16', fp16: 'FP16', fp8: 'FP8', mxfp4: 'MXFP4' };

const layerTypes = (m) => m.layerTypes || Array(m.layers).fill('full');
const isMoE = (m) => m.ffn.kind === 'moe';

// ---------------- parameters ----------------

// One linear layer y = x·W (+ b) for an MLP: gated (SwiGLU: W_gate, W_up, W_down) or plain (W_1, W_2).
const mlpParams = (H, F, gated, bias) => (gated ? 3 * H * F + (bias ? 2 * F + H : 0) : 2 * H * F + (bias ? F + H : 0));
const normParams = (m) => (m.norm === 'LayerNorm' ? 2 : 1) * m.hidden;   // LayerNorm has a scale and a bias

// Token-mixing parameters of one layer of the given type.
function attnParams(m, type) {
  const H = m.hidden;
  if (type === 'linear') {
    const g = m.linearAttention, kT = g.keyHeads * g.keyDim, vT = g.valueHeads * g.valueDim;
    return H * (2 * kT + 2 * vT)     // in_proj: q, k, v and the output gate z
      + H * 2 * g.valueHeads         // in_proj: per value head, write strength (b) and decay input (a)
      + (2 * kT + vT) * g.conv       // depthwise causal convolution over q, k, v
      + 2 * g.valueHeads             // A_log and dt_bias (decay parameters)
      + g.valueDim                   // RMSNorm on each head's output, weights shared across heads
      + vT * H;                      // out_proj
  }
  const a = m.attention;
  if (a.kind === 'MLA') {
    const dqk = a.nopeDim + a.ropeDim;
    return H * a.qRank + a.qRank + a.qRank * a.heads * dqk                       // W_DQ, its norm, W_UQ
      + H * (a.kvRank + a.ropeDim) + a.kvRank + a.kvRank * a.heads * (a.nopeDim + a.vDim)   // W_DKV + W_KR, norm, W_UK and W_UV
      + a.heads * a.vDim * H;                                                     // W_O
  }
  const q = a.heads * a.headDim, k = a.kvHeads * a.headDim, qOut = q * (a.outputGate ? 2 : 1);
  let p = H * qOut + 2 * H * k + q * H;
  if (a.bias === 'qkv') p += qOut + 2 * k;
  if (a.bias === 'all') p += qOut + 2 * k + H;
  if (a.qkNorm) p += 2 * a.headDim;
  if (a.sinks) p += a.heads;
  return p;
}

// One MoE layer: router, routed experts, shared experts.
function moeLayer(m) {
  const f = m.ffn, H = m.hidden;
  const perExpert = mlpParams(H, f.expertInter, f.gated, f.expertBias);
  const router = H * f.experts + (f.routerBias ? f.experts : 0);
  const sharedOne = f.shared ? mlpParams(H, f.sharedInter ?? f.expertInter, f.gated, false) : 0;
  const shared = f.shared * sharedOne + (f.sharedGate ? H : 0);
  return { perExpert, router, routed: f.experts * perExpert, shared, total: router + f.experts * perExpert + shared,
    active: router + f.topK * perExpert + shared };
}

// ViT encoder: patch embedding, position embedding, blocks, and the merger that maps groups of
// merge × merge patches to one token of the language model's width (plus DeepStack mergers, if any).
function visionParams(v) {
  if (!v) return 0;
  const h = v.hidden, f = v.inter, hm = h * v.merge * v.merge;
  const block = 2 * (2 * h) + (h * 3 * h + 3 * h) + (h * h + h) + (h * f + f) + (f * h + h);
  const patch = v.channels * v.temporalPatch * v.patch * v.patch * h + h;
  const merger = (postNorm) => 2 * (postNorm ? hm : h) + (hm * hm + hm) + (hm * v.outHidden + v.outHidden);
  return v.layers * block + patch + v.posEmbed * h + merger(false) + v.deepstack * merger(true);
}

function params(m) {
  const H = m.hidden, f = m.ffn, types = layerTypes(m);
  let attn = 0;
  const attnByType = {};
  types.forEach((t) => { const p = attnParams(m, t); attn += p; attnByType[t] = (attnByType[t] || 0) + p; });
  const norms = 2 * m.layers * normParams(m);
  let denseFFN = 0, router = 0, routed = 0, shared = 0, activeRouted = 0, perExpert = 0, denseLayers = m.layers, moeLayers = 0;
  if (isMoE(m)) {
    const ml = moeLayer(m);
    denseLayers = f.denseLayers; moeLayers = m.layers - f.denseLayers; perExpert = ml.perExpert;
    denseFFN = denseLayers ? denseLayers * mlpParams(H, f.denseInter, f.gated, false) : 0;
    router = moeLayers * ml.router; routed = moeLayers * ml.routed; shared = moeLayers * ml.shared;
    activeRouted = moeLayers * f.topK * ml.perExpert;
  } else denseFFN = m.layers * mlpParams(H, f.inter, f.gated, f.bias);
  const embed = m.vocab * H, pos = m.position === 'learned' ? m.context * H : 0, head = m.tied || m.noHead ? 0 : m.vocab * H;
  const finalNorm = normParams(m);
  const language = embed + pos + head + finalNorm + attn + norms + denseFFN + router + routed + shared;
  const vision = visionParams(m.vision);
  let mtp = 0;
  if (m.mtp) {
    const layer = attnParams(m, 'full') + (isMoE(m) ? moeLayer(m).total : mlpParams(H, f.inter, f.gated, f.bias)) + 2 * normParams(m);
    mtp = m.mtp.layers * (2 * H * H + 3 * normParams(m) + layer) + (m.mtp.ownEmbedding ? embed + head : 0);
  }
  const active = language - routed + activeRouted;       // one text token: unselected experts and the vision encoder unused
  return {
    embed, pos, head, finalNorm, attn, attnByType, norms, denseFFN, router, routed, shared, perExpert,
    denseLayers, moeLayers, language, vision, mtp,
    total: language + vision,                             // the headline count: language model + vision encoder
    checkpoint: language + vision + mtp,                  // everything the published checkpoint stores
    active,
    matmul: active - (m.tied ? 0 : embed) - pos,          // weights multiplied per token: the N in 2·N FLOPs per token
  };
}

// ---------------- weights in memory ----------------

// Parameters by tensor class. A released checkpoint (models.js `release`) assigns one dtype per class.
function weightClasses(m) {
  const P = params(m);
  return {
    experts: P.routed,                                   // routed experts
    linear: P.attn + P.shared + P.denseFFN,              // attention projections, shared experts, dense FFNs
    embed: P.embed + P.head + P.pos,                     // embedding table and output layer
    small: P.norms + P.finalNorm + P.router,             // norms and routers
    vision: P.vision,
  };
}
const CLASS_NAME = { experts: 'routed experts', linear: 'attention, shared experts and dense FFNs', embed: 'embedding and output layer', small: 'norms and routers', vision: 'vision encoder' };

// Dtype per class for the published weights; classes a release doesn't name stay in BF16.
// Serving on G GPUs with r per node: tensor parallelism inside a node, pipeline parallelism across nodes
// (vLLM's guidance). TP must divide the attention heads; TP × PP can then use fewer than G GPUs.
function servingLayout(m, G, r) {
  let tp = Math.min(G, r);
  while (m.attention.heads % tp) tp--;
  const pp = Math.max(1, Math.floor(G / tp));
  return { tp, pp, used: tp * pp, why: tp < Math.min(G, r) ? `the tensor-parallel size must divide the ${m.attention.heads} attention heads` : `${G} is not a multiple of ${r} GPUs per node` };
}

function releaseDtypes(m) {
  const r = m.release || {};
  const d = { experts: 'bf16', linear: 'bf16', embed: 'bf16', small: 'bf16', vision: 'bf16' };
  if (r.all) Object.keys(d).forEach((k) => { d[k] = r.all; });
  Object.keys(r).forEach((k) => { if (k !== 'all') d[k] = r[k]; });
  return d;
}

function weightBytes(m, dtypes) {
  const c = weightClasses(m), d = { ...releaseDtypes(m), ...dtypes };
  let total = 0;
  const byClass = {};
  Object.keys(c).forEach((k) => { byClass[k] = c[k] * DTYPE_BYTES[d[k]]; total += byClass[k]; });
  return { total, byClass, dtypes: d };
}

// ---------------- mixture of experts ----------------

// Expected number of distinct experts selected by n tokens when each token selects k of E uniformly.
const expertsTouched = (E, k, n) => E * (1 - Math.pow(1 - k / E, n));

// Bytes of one token's hidden vector sent to its k routed experts in one MoE layer (the same returns).
const dispatchBytes = (m, actBytes = 2) => m.ffn.topK * m.hidden * actBytes;

// ---------------- KV cache and recurrent state ----------------

// Values one sequence keeps after ctx positions. Full attention: K and V (or the MLA latent) for every
// position. Sliding window: at most `window` positions. Linear attention: a fixed state per value head
// (keyDim × valueDim) plus the last conv − 1 inputs of the causal convolution.
function cacheValues(m, ctx) {
  const a = m.attention;
  const perPos = a.kind === 'MLA' ? a.kvRank + a.ropeDim : 2 * a.kvHeads * a.headDim;
  let kv = 0, state = 0, conv = 0;
  layerTypes(m).forEach((t) => {
    if (t === 'full') kv += perPos * ctx;
    else if (t === 'sliding') kv += perPos * Math.min(ctx, a.window);
    else {
      const g = m.linearAttention;
      state += g.valueHeads * g.keyDim * g.valueDim;
      conv += (g.conv - 1) * (2 * g.keyHeads * g.keyDim + g.valueHeads * g.valueDim);
    }
  });
  return { kv, state, conv, perPos };
}

// Bytes per sequence. The linear-attention state uses its own dtype (Qwen3.5: FP32, `mamba_ssm_dtype`).
function cacheBytes(m, ctx, kvDtype = 'bf16') {
  const v = cacheValues(m, ctx), b = DTYPE_BYTES[kvDtype];
  return { kv: v.kv * b, state: v.state * (m.linearAttention?.stateBytes ?? 4) + v.conv * b,
    get total() { return this.kv + this.state; } };
}

// Bytes added per position once the context is longer than any sliding window (full-attention layers).
function kvBytesPerToken(m, kvDtype = 'bf16') {
  const full = layerTypes(m).filter((t) => t === 'full').length;
  return full * cacheValues(m, 1).perPos * DTYPE_BYTES[kvDtype];
}

// Per GPU under tensor parallelism: K/V heads are split over TP (a head is replicated when TP > heads);
// the MLA latent is replicated on every TP rank; linear-attention value heads are split over TP.
function kvBytesPerTokenPerGPU(m, tp, kvDtype = 'bf16') {
  const a = m.attention, full = layerTypes(m).filter((t) => t === 'full').length;
  const perPos = a.kind === 'MLA' ? a.kvRank + a.ropeDim : 2 * Math.ceil(a.kvHeads / tp) * a.headDim;
  return full * perPos * DTYPE_BYTES[kvDtype];
}

// Per GPU, sequence of ctx positions, with K/V heads and linear-attention heads split over TP.
function cacheBytesPerGPU(m, ctx, tp = 1, kvDtype = 'bf16') {
  const a = m.attention, b = DTYPE_BYTES[kvDtype];
  const perPos = a.kind === 'MLA' ? a.kvRank + a.ropeDim : 2 * Math.ceil(a.kvHeads / tp) * a.headDim;
  let bytes = 0;
  layerTypes(m).forEach((t) => {
    if (t === 'full') bytes += perPos * ctx * b;
    else if (t === 'sliding') bytes += perPos * Math.min(ctx, a.window) * b;
    else {
      const g = m.linearAttention, vh = Math.ceil(g.valueHeads / tp), kh = Math.ceil(g.keyHeads / tp);
      bytes += vh * g.keyDim * g.valueDim * g.stateBytes + (g.conv - 1) * (2 * kh * g.keyDim + vh * g.valueDim) * b;
    }
  });
  return bytes;
}

// ---------------- FLOPs ----------------

// Query/key and value widths per head of softmax attention.
function attnDims(m) {
  const a = m.attention;
  return a.kind === 'MLA' ? { heads: a.heads, dqk: a.nopeDim + a.ropeDim, dv: a.vDim } : { heads: a.heads, dqk: a.headDim, dv: a.headDim };
}

// Attention FLOPs for one token in one layer, beyond the weight multiplications (scores Q·Kᵀ and A·V).
// mode 'causal': averaged over a causal sequence of length T (training and prefill); a token at
//   position t attends to t + 1 positions, T/2 on average (sliding window: at most `window`).
// mode 'decode': one new token attending to T cached positions.
// MLA in decode runs absorbed: scores against the (kvRank + ropeDim)-value latent and values from the
//   kvRank-value latent, so the width per head is 576 + 512 instead of 192 + 128.
// Linear attention (Gated DeltaNet): ≈ 8·d_k·d_v per value head per token for the state update and
//   readout, independent of T (approximate).
function attnFlopsPerToken(m, type, T, mode) {
  if (type === 'linear') { const g = m.linearAttention; return 8 * g.valueHeads * g.keyDim * g.valueDim; }
  const a = m.attention, d = attnDims(m), w = type === 'sliding' ? a.window : Infinity;
  const positions = mode === 'decode' ? Math.min(T, w) : (T > w ? w - (w * w) / (2 * T) : T / 2);
  const width = mode === 'decode' && a.kind === 'MLA' ? 2 * a.kvRank + a.ropeDim : d.dqk + d.dv;
  return 2 * d.heads * positions * width;
}

// Forward FLOPs per token: 2 per multiplied weight (multiply + add) plus attention, over all layers.
function forwardFlopsPerToken(m, T, mode = 'causal') {
  const P = params(m), attention = layerTypes(m).reduce((s, t) => s + attnFlopsPerToken(m, t, T, mode), 0);
  return { weights: 2 * P.matmul, attention, total: 2 * P.matmul + attention };
}

// Training: forward + backward (twice the forward FLOPs) = 3 forward passes; full activation
// recomputation runs the forward pass once more. 3 × 2·N = the usual 6·N FLOPs per token.
function trainingFlops(m, tokens, seq, recompute = 'none') {
  const fwd = forwardFlopsPerToken(m, seq, 'causal'), passes = recompute === 'full' ? 4 : 3;
  return { fwd, passes, perToken: passes * fwd.total, total: passes * fwd.total * tokens };
}

// ---------------- training memory ----------------

// Bytes per parameter of the training state, mixed precision as in Megatron-LM: BF16 weights,
// FP32 gradients, FP32 master weights, optimizer states (AdamW: m and v; Muon: one momentum buffer).
// DeepSeek-V3 keeps AdamW moments in BF16 (`trainPrecision.moments`).
function trainBytesPerParam(m, optimizer = 'adamw') {
  const mom = m.trainPrecision?.moments === 'bf16' ? 2 : 4, states = optimizer === 'muon' ? 1 : 2;
  return { weights: 2, grads: 4, master: 4, optimizer: states * mom, states, momentDtype: mom === 2 ? 'bf16' : 'fp32',
    total: 2 + 4 + 4 + states * mom };
}

// Values saved for the backward pass per token, summed over layers (stored in BF16). Approximate:
// the exact set depends on the framework's kernels. FlashAttention-style kernels store no T × T scores.
// Full recomputation keeps only each layer's input (H values per layer) and recomputes the rest.
function savedValuesPerToken(m, recompute = 'none') {
  const H = m.hidden, f = m.ffn, a = m.attention, d = attnDims(m);
  if (recompute === 'full') return m.layers * H;
  const ffnVals = (F) => (f.gated ? 4 : 2) * F;   // gated: gate, up, activation, product; plain: pre- and post-activation
  const mix = (t) => {
    if (t === 'linear') {
      const g = m.linearAttention, kT = g.keyHeads * g.keyDim, vT = g.valueHeads * g.valueDim;
      return 2 * H + 2 * kT + 3 * vT + 2 * g.valueHeads;   // norm in/out, q, k, v, gate z, output, b and a
    }
    if (a.kind === 'MLA') return 2 * H + a.qRank + 2 * d.heads * d.dqk + (a.kvRank + a.ropeDim) + 2 * d.heads * d.dv;
    return 2 * H + 2 * d.heads * d.dqk * (a.outputGate ? 1.5 : 1) + 2 * a.kvHeads * d.dqk;   // q and output (+ gate), k, v
  };
  const P = params(m);
  const ffn = !isMoE(m) ? 2 * H + ffnVals(f.inter)
    : (P.denseLayers * (2 * H + ffnVals(f.denseInter || 0))
      + P.moeLayers * (2 * H + f.experts + (f.topK + f.shared) * ffnVals(f.expertInter) + 2 * f.topK * H)) / m.layers;
  return layerTypes(m).reduce((s, t) => s + mix(t) + ffn, 0);
}

// ---------------- inference on one serving instance ----------------
// An instance of G GPUs holds one copy of the weights, split evenly (TP, PP and EP combined), and
// the caches of its sequences. Totals are computed for the instance; per GPU = total / G, so each GPU
// reads only the weights it holds (the expert-parallel case included).
// Decode step: B sequences, one new token each, at context c. Prefill: B sequences of T tokens.
function inferenceStep(m, { mode = 'decode', B = 1, c = 4096, T = 4096, dtypes = {}, kvDtype = 'bf16' } = {}) {
  const P = params(m), W = weightBytes(m, dtypes), f = m.ffn, d = releaseDtypes(m);
  const embedTable = m.tied ? 0 : P.embed * DTYPE_BYTES[{ ...d, ...dtypes }.embed];     // a lookup reads B rows only
  const tokens = mode === 'decode' ? B : B * T;
  const expertsRead = isMoE(m) ? (W.byClass.experts * expertsTouched(f.experts, f.topK, tokens)) / f.experts : 0;
  const weightsRead = W.total - W.byClass.vision - embedTable - W.byClass.experts + expertsRead;
  const headF = 2 * m.hidden * m.vocab;
  if (mode === 'decode') {
    const cache = cacheBytes(m, c, kvDtype), fl = B * forwardFlopsPerToken(m, c, 'decode').total;
    const bytes = weightsRead + B * (cache.kv + 2 * cache.state);   // read every cached position; read and write the fixed state
    return { flops: fl, bytes, weightsRead, cacheRead: B * (cache.kv + 2 * cache.state), intensity: fl / bytes, tokens: B };
  }
  const fl = B * T * forwardFlopsPerToken(m, T, 'causal').total - B * (T - 1) * headF;   // logits for the last position only
  const write = B * cacheBytes(m, T, kvDtype).total;
  return { flops: fl, bytes: weightsRead + write, weightsRead, cacheWrite: write, intensity: fl / (weightsRead + write), tokens: B * T };
}

// Time lower bound when the reader gives per-GPU peak FLOP/s and memory bandwidth (B/s): the step
// takes at least the longer of compute time and memory time, with no overlap losses.
function stepTime(step, G, flopsPerGPU, bwPerGPU) {
  if (!(flopsPerGPU > 0) || !(bwPerGPU > 0)) return null;
  const tc = step.flops / (G * flopsPerGPU), tm = step.bytes / (G * bwPerGPU);
  return { tc, tm, t: Math.max(tc, tm), bound: tc >= tm ? 'compute' : 'memory bandwidth' };
}

// ---------------- parallel groups and collectives ----------------

// Ranks are numbered with TP innermost, then CP, DP, PP; each node holds gpusPerNode consecutive ranks.
// EP groups are taken across DP × CP (stride TP) when they fit there, otherwise contiguously.
// Returns the group size n, ranks of one group per node r, and nodes the group spans.
function placeGroup(n, stride, perNode) {
  const r = Math.max(1, Math.min(n, Math.floor(perNode / stride)));
  return { n, r, nodes: Math.ceil(n / r) };
}
function groupPlacement(S, dim) {
  const epInDp = S.ep <= S.dp * S.cp;
  const g = {
    tp: [S.tp, 1], cp: [S.cp, S.tp], dp: [S.dp, S.tp * S.cp], pp: [S.pp, S.tp * S.cp * S.dp],
    ep: [S.ep, epInDp ? S.tp : 1],
    dpcp: [S.dp * S.cp, S.tp],                                            // ranks with identical dense shards
    edp: [(S.dp * S.cp * S.tp) / (S.ep > 1 ? S.ep : S.tp), epInDp ? S.tp * S.ep : S.ep],   // ranks with identical expert shards
  }[dim];
  return placeGroup(g[0], g[1], S.gpusPerNode);
}

// ---------------- training split over GPUs ----------------
// S: { dp, tp, pp, ep, cp, gpusPerNode, zero, recompute, seq, mbs, gbs, optimizer }.
// Experts are split over EP when ep > 1, otherwise TP-split like dense weights.
function trainingLayout(m, S) {
  const P = params(m), g = isMoE(m) && S.ep > 1 ? S.ep : S.tp, N = S.dp * S.tp * S.pp * S.cp, micro = S.gbs / (S.dp * S.mbs);
  return {
    N, nodes: Math.ceil(N / S.gpusPerNode), g,
    denseShard: (P.total - P.routed) / (S.tp * S.pp),       // parameters per GPU outside routed experts
    expertShard: P.routed / (S.pp * g),                     // routed-expert parameters per GPU
    dpDense: S.dp * S.cp, dpExpert: (S.dp * S.cp * S.tp) / g,
    layersPerStage: Math.ceil(m.layers / S.pp), moeLayersPerStage: Math.ceil(P.moeLayers / S.pp),
    micro, inflight: Math.min(S.pp, micro), tokensMicro: (S.mbs * S.seq) / S.cp,
  };
}

function validateLayout(m, S) {
  const out = [], a = m.attention, L = trainingLayout(m, S);
  if (a.heads % S.tp) out.push(`TP ${S.tp} does not divide the ${a.heads} query heads.`);
  if (a.kind !== 'MLA' && S.tp > a.kvHeads) out.push(`TP ${S.tp} is larger than the ${a.kvHeads} K/V heads: each K/V head is replicated on ${S.tp / a.kvHeads} GPUs.`);
  if (S.tp > S.gpusPerNode) out.push(`TP ${S.tp} spans ${Math.ceil(S.tp / S.gpusPerNode)} nodes: its collectives, two per layer in each direction, cross the network.`);
  if (S.pp > m.layers) out.push(`PP ${S.pp} exceeds the ${m.layers} layers.`);
  if (isMoE(m)) {
    if (m.ffn.experts % S.ep) out.push(`EP ${S.ep} does not divide the ${m.ffn.experts} experts.`);
    if ((S.dp * S.cp * S.tp) % S.ep) out.push(`EP ${S.ep} must divide DP × CP × TP = ${S.dp * S.cp * S.tp}.`);
  } else if (S.ep > 1) out.push('EP applies only to MoE models.');
  if (S.cp > 1 && S.seq % (2 * S.cp)) out.push(`With CP ${S.cp}, the sequence length ${S.seq} must be divisible by 2 × CP = ${2 * S.cp}: each CP rank takes two chunks.`);
  if (!Number.isInteger(L.micro) || L.micro < 1) out.push(`The global batch ${S.gbs} is not a multiple of DP × micro-batch = ${S.dp * S.mbs}.`);
  return out;
}

// Training memory per GPU (the first pipeline stage, which holds the most activations).
function trainingMemoryPerGPU(m, S) {
  const L = trainingLayout(m, S), b = trainBytesPerParam(m, S.optimizer), z = S.zero;
  const shard = (zMin) => L.denseShard / (z >= zMin ? L.dpDense : 1) + L.expertShard / (z >= zMin ? L.dpExpert : 1);
  const perLayer = savedValuesPerToken(m, S.recompute) / m.layers;
  const rows = [
    { name: 'Weights, BF16', kind: 'w', bytes: shard(3) * b.weights, note: z >= 3 ? 'sharded over data parallelism (ZeRO-3)' : 'a full copy of this GPU\'s TP/PP/EP shard' },
    { name: 'Gradients, FP32', kind: 'g', bytes: shard(2) * b.grads, note: z >= 2 ? 'sharded (ZeRO-2 and above)' : 'not sharded' },
    { name: 'Master weights, FP32', kind: 'g-mid', bytes: shard(1) * b.master, note: z >= 1 ? 'sharded (ZeRO-1 and above)' : 'not sharded' },
    { name: `Optimizer state, ${b.momentDtype.toUpperCase()}`, kind: 'g-dark', bytes: shard(1) * b.optimizer, note: z >= 1 ? 'sharded (ZeRO-1 and above)' : 'not sharded' },
    { name: 'Saved activations', kind: 'a', approx: true, bytes: (L.layersPerStage * L.inflight * L.tokensMicro * perLayer * 2) / S.tp,
      note: `${L.layersPerStage} layers × ${L.inflight} micro-batch${L.inflight > 1 ? 'es' : ''} in flight × ${fmtN(L.tokensMicro)} tokens, ÷ TP` },
    { name: 'Logits for the loss, FP32', kind: 'a', approx: true, bytes: (L.tokensMicro * m.vocab * 4) / S.tp, note: 'last pipeline stage, one micro-batch' },
  ];
  if (z >= 3) rows.push({ name: 'Gathered weights of one layer', kind: 'w', approx: true, bytes: ((2 * params(m).total) / m.layers / S.tp) * 2, note: 'ZeRO-3: the current and the next layer' });
  return rows;
}

// Collectives per GPU per training step. The forward pass runs twice with full recomputation, so
// its collectives (TP, EP, CP forward passes, ZeRO-3 gathers) run twice.
function trainingCollectiveRows(m, S) {
  const L = trainingLayout(m, S), H = m.hidden, b = trainBytesPerParam(m, S.optimizer), out = [], fwd = S.recompute === 'full' ? 2 : 1;
  const act = L.tokensMicro * H * 2;
  const add = (dim, op, when, M, count, place) => {
    const pl = place || groupPlacement(S, dim), per = collectiveBytes(op, pl.n, pl.r, M);
    if (pl.n > 1 && count > 0) out.push({ dim, op, when, M, count, n: pl.n, r: pl.r, nodes: pl.nodes, intra: per.intra * count, inter: per.inter * count });
  };
  add('tp', 'all-reduce', 'after the attention output and FFN down projections (with sequence parallelism: a reduce-scatter and an all-gather of the same size)', act, (2 * fwd + 2) * L.layersPerStage * L.micro);
  if (S.cp > 1) {
    const a = m.attention, perPos = a.kind === 'MLA' ? a.kvRank + a.ropeDim : (2 * a.kvHeads * a.headDim) / Math.min(S.tp, a.kvHeads);
    add('cp', 'p2p', 'inside attention: K and V blocks passed around the ring (forward), plus their gradients (backward)', L.tokensMicro * perPos * 2, (S.cp - 1) * (fwd + 2) * L.layersPerStage * L.micro);
  }
  add('pp', 'p2p', 'at each stage boundary: activations forward, their gradients backward', act / S.tp, 2 * L.micro);
  if (isMoE(m) && S.ep > 1) add('ep', 'all-to-all', 'in every MoE layer: tokens to their experts and back (dispatch and combine)', (L.tokensMicro / S.tp) * m.ffn.topK * H * 2, (2 * fwd + 2) * L.moeLayersPerStage * L.micro);
  const dpRows = (dim, n, shard, label) => {
    if (n <= 1 || shard <= 0) return;
    const G = shard * b.grads, Wt = shard * b.weights;
    if (S.zero === 0) add(dim, 'all-reduce', `${label}gradients, once per step`, G, 1);
    else if (S.zero < 3) { add(dim, 'reduce-scatter', `${label}gradients: each GPU keeps 1/${n}`, G, 1); add(dim, 'all-gather', `${label}updated BF16 weights after the optimizer step`, Wt, 1); }
    else { add(dim, 'all-gather', `${label}weights before each layer is used, forward and backward, every micro-batch`, Wt, (fwd + 1) * L.micro); add(dim, 'reduce-scatter', `${label}gradients`, G, 1); }
  };
  dpRows('dpcp', L.dpDense, L.denseShard, isMoE(m) ? 'non-expert ' : '');
  if (isMoE(m)) dpRows(S.ep > 1 ? 'edp' : 'dpcp', L.dpExpert, L.expertShard, 'expert ');
  return out;
}

// Bytes one GPU sends for one collective over n ranks, r per node, on a buffer of M bytes.
// Two levels: inside the node (r ranks), then across nodes between ranks with the same local index.
// All-gather and reduce-scatter: M is the full (gathered) buffer; each rank contributes M/n.
// Ring volumes: all-gather (r−1)/r·M inside, (N−1)/N·M/r across; all-reduce twice that.
// All-to-all: each rank sends (n−1)/n·M, of which (n−r)/n·M leaves the node.
function collectiveBytes(op, n, r, M) {
  const N = Math.ceil(n / r);
  if (n <= 1) return { intra: 0, inter: 0 };
  if (op === 'all-reduce') return { intra: (2 * (r - 1) / r) * M, inter: N > 1 ? (2 * (N - 1) / N) * M / r : 0 };
  if (op === 'all-gather' || op === 'reduce-scatter') return { intra: ((r - 1) / r) * M, inter: N > 1 ? ((N - 1) / N) * M / r : 0 };
  if (op === 'all-to-all') return { intra: ((r - 1) / n) * M, inter: ((n - r) / n) * M };
  if (op === 'p2p') return N > 1 ? { intra: 0, inter: M } : { intra: M, inter: 0 };
  throw new Error(`unknown collective ${op}`);
}

// ---------------- formatting (shared with the chapters) ----------------

function fmtN(n) { return Math.round(n).toLocaleString('en-US'); }
// Three significant digits at most, without trailing zeros: 80 GB, 1.5 GB, 20.8 GB, 1.02 GB.
function fmtSig(x) {
  const s = x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}
function fmtBytes(b) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']; let i = 0;
  while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return i === 0 ? `${Math.round(b)} B` : `${fmtSig(b)} ${u[i]}`;
}
function fmtParams(n) {
  if (n >= 1e12) return `${fmtSig(n / 1e12)}T`;
  if (n >= 1e9) return `${fmtSig(n / 1e9)}B`;
  if (n >= 1e6) return `${fmtSig(n / 1e6)}M`;
  if (n >= 1e3) return `${fmtSig(n / 1e3)}K`;
  return `${n}`;
}
function fmtFlops(n) {
  for (const [s, u] of [[1e18, 'EFLOP'], [1e15, 'PFLOP'], [1e12, 'TFLOP'], [1e9, 'GFLOP'], [1e6, 'MFLOP']]) if (n >= s) return `${fmtSig(n / s)} ${u}`;
  return `${fmtN(n)} FLOP`;
}
// 3.78 × 10^25, with a superscript exponent (HTML).
function fmtExp(x) { const [mant, exp] = x.toExponential(2).split('e'); return `${mant} × 10<sup>${Number(exp)}</sup>`; }
// The same for SVG text, with Unicode superscript digits: 3.78 × 10²⁵; exact powers of ten as 10²⁵.
function fmtExpText(x) {
  const [mant, exp] = x.toExponential(2).split('e'), sup = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const e = String(Number(exp)).split('').map((c) => (c === '-' ? '⁻' : sup[Number(c)])).join('');
  return `${Number(mant) === 1 ? '' : `${mant} × `}10${e}`;
}
function fmtPct(a, b) {
  const p = (100 * a) / b;
  return p > 0 && p < 0.1 ? '< 0.1%' : `${p >= 10 ? p.toFixed(0) : p.toFixed(1)}%`;
}
