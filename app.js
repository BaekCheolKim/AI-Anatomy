// Template: builds the diagram and every explanation from the selected model in models.js.
// Text branches on architecture features (attention kind, FFN kind, position, norm), never on model name.
// The example values (token IDs, attention weights, routing, probabilities) are ILLUSTRATIVE.

const TOKENS = ['The', ' cat', ' sat', ' on', ' the'];
const IDS = [791, 8415, 7731, 389, 279];
const NEXT = [[' mat', 0.41], [' floor', 0.17], [' couch', 0.08], [' bed', 0.06], [' sofa', 0.05]];
const T = TOKENS.length;

// ---------- formatting ----------
const fmt = (n) => n.toLocaleString('en-US');
const fmtB = (n) => (n >= 100e9 ? (n / 1e9).toFixed(0) : (n / 1e9).toFixed(2)) + 'B';
const fmtM = (n) => (n >= 1e9 ? fmtB(n) : (n / 1e6).toFixed(n >= 100e6 ? 0 : 1) + 'M');
const fmtBytes = (n) => (n >= 1024 ** 2 ? (n / 1024 ** 2).toFixed(2) + ' MiB' : (n / 1024).toFixed(n % 1024 ? 1 : 0) + ' KiB');
const pct = (a, b) => ((100 * a) / b).toFixed(0) + '%';
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const show = (t) => esc(t.replace(/^ /, '␣'));

// ---------- numbers derived from the config ----------
function params(m) {
  const H = m.hidden, L = m.layers, a = m.attention, f = m.ffn;
  let attn;
  if (a.kind === 'MLA') {
    const dqk = a.nopeDim + a.ropeDim;
    attn = H * a.qRank + a.qRank + a.qRank * a.heads * dqk          // W_DQ, q norm, W_UQ
      + H * (a.kvRank + a.ropeDim) + a.kvRank                         // W_DKV + W_KR, kv norm
      + a.kvRank * a.heads * (a.nopeDim + a.vDim)                     // W_UK, W_UV
      + a.heads * a.vDim * H;                                          // W_O
  } else {
    const q = a.heads * a.headDim, k = kvHeads(m) * a.headDim;
    attn = 2 * H * q + 2 * H * k + (a.bias === 'qkv' ? q + 2 * k : a.bias === 'all' ? q + 2 * k + H : 0);
  }
  const mlp = (F, gated, bias) => (gated ? 3 * H * F + (bias ? 2 * F + H : 0) : 2 * H * F + (bias ? F + H : 0));
  let ffnTotal, ffnActive, perExpert = 0, ffnLayer;
  if (f.kind === 'dense') {
    ffnLayer = mlp(f.inter, f.gated, f.bias);
    ffnTotal = ffnActive = L * ffnLayer;
  } else {
    const moeLayers = L - f.denseLayers;
    perExpert = mlp(f.expertInter, f.gated, false);
    const router = f.experts * H + (f.routerBias ? f.experts : 0);
    const dense = f.denseLayers ? f.denseLayers * mlp(f.denseInter, f.gated, false) : 0;
    ffnLayer = router + (f.experts + f.shared) * perExpert;
    ffnTotal = dense + moeLayers * ffnLayer;
    ffnActive = dense + moeLayers * (router + (f.topK + f.shared) * perExpert);
  }
  const embed = m.vocab * H;
  const pos = m.position === 'learned' ? m.context * H : 0;
  const head = m.tied ? 0 : m.vocab * H;
  const norms = (2 * L + 1) * (m.norm === 'LayerNorm' ? 2 * H : H);
  const total = embed + pos + head + L * attn + ffnTotal + norms;
  return { embed, pos, head, attnLayer: attn, ffnLayer, ffnTotal, perExpert, norms, total, active: total - ffnTotal + ffnActive };
}
function kvHeads(m) { return m.attention.kind === 'MHA' ? m.attention.heads : m.attention.kvHeads; }
function kvValuesPerToken(m) {   // cached values per token, all layers
  const a = m.attention;
  return a.kind === 'MLA' ? m.layers * (a.kvRank + a.ropeDim) : 2 * m.layers * kvHeads(m) * a.headDim;
}
const actName = (m) => (m.ffn.act === 'silu' ? 'SiLU' : 'GELU');
const ffnName = (m) => (m.ffn.gated ? (m.ffn.act === 'silu' ? 'SwiGLU' : 'GeGLU') : `${actName(m)} MLP`);
const normName = (m) => m.norm;

// ---------- small visuals ----------
function attentionGrid() {
  const raw = [[0, 0, 0, 0, 0], [1.0, 1.5, 0, 0, 0], [0.3, 2.2, 1.0, 0, 0], [0.2, 1.0, 1.8, 1.2, 0], [0.1, 2.0, 0.8, 1.6, 0.9]];
  const w = raw.map((row, i) => {
    const e = row.map((s, j) => (j <= i ? Math.exp(s) : 0));
    const sum = e.reduce((x, y) => x + y, 0);
    return e.map((x) => x / sum);
  });
  const cell = 36, left = 44, top = 30;
  let svg = `<svg width="${left + cell * T + 4}" height="${top + cell * T + 4}" role="img" aria-label="attention weights">`;
  svg += `<text x="${left}" y="10" font-size="10" fill="#6b7280">key position j →</text>`;
  TOKENS.forEach((t, j) => { svg += `<text x="${left + j * cell + cell / 2}" y="${top - 6}" font-size="11" text-anchor="middle" fill="#4b5563">${show(t)}</text>`; });
  TOKENS.forEach((t, i) => {
    svg += `<text x="${left - 6}" y="${top + i * cell + cell / 2 + 4}" font-size="11" text-anchor="end" fill="#4b5563">${show(t)}</text>`;
    w[i].forEach((v, j) => {
      const masked = j > i;
      const fill = masked ? '#eef0f3' : `rgba(37, 99, 235, ${0.08 + 0.92 * v})`;
      const tip = masked ? `A[${i},${j}] = 0 (causal mask, j > i)` : `A[${i},${j}] = ${v.toFixed(2)}  (query "${t.trim()}", key "${TOKENS[j].trim()}")`;
      svg += `<rect x="${left + j * cell + 1}" y="${top + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="${fill}"><title>${esc(tip)}</title></rect>`;
    });
  });
  return `<div class="viz-row">${svg}<div class="note">A = softmax(QKᵀ/√d_h) for one head.<br>Row i = query position i; each row sums to 1.<br>Grey: masked (j &gt; i). Hover a cell for its value.</div></div>`;
}

function activationChart(m) {
  const W = 220, H = 110, x0 = -5, x1 = 4, y0 = -1, y1 = 4;
  const sx = (x) => ((x - x0) / (x1 - x0)) * W, sy = (y) => H - ((y - y0) / (y1 - y0)) * H;
  const fn = m.ffn.act === 'silu' ? (x) => x / (1 + Math.exp(-x))
    : (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  const pts = [];
  for (let x = x0; x <= x1 + 1e-9; x += 0.1) pts.push(`${sx(x).toFixed(1)},${sy(fn(x)).toFixed(1)}`);
  return `<div class="viz-row"><svg width="${W}" height="${H}" role="img" aria-label="${actName(m)} curve">
      <line x1="0" y1="${sy(0)}" x2="${W}" y2="${sy(0)}" stroke="#d5dae1"/>
      <line x1="${sx(0)}" y1="0" x2="${sx(0)}" y2="${H}" stroke="#d5dae1"/>
      <line x1="${sx(-1)}" y1="${sy(-1)}" x2="${sx(4)}" y2="${sy(4)}" stroke="#b9c0ca" stroke-dasharray="3 3"/>
      <polyline points="${pts.join(' ')}" fill="none" stroke="#2563eb" stroke-width="2"/></svg>
    <div class="note">Solid: ${actName(m)}(x).<br>Dashed: y = x (identity).</div></div>`;
}

function routerGrid(m) {
  const f = m.ffn, size = f.experts > 32 ? 12 : 26;
  // One grid row per routing group (or a single row). The illustrative selection obeys group-limited routing.
  const groups = f.groups || 1, perGroup = f.experts / groups, cols = Math.min(perGroup, 32);
  const chosenGroups = Array.from({ length: f.topkGroups || 1 }, (_, i) => (i * 3 + 1) % groups);
  const picked = new Set(Array.from({ length: f.topK }, (_, i) =>
    chosenGroups[i % chosenGroups.length] * perGroup + ((i * 7 + 3) % perGroup)));
  let cells = '';
  for (let e = 0; e < f.experts; e++) {
    const on = picked.has(e);
    cells += `<span class="ex${on ? ' on' : ''}" style="width:${size}px;height:${size}px" title="expert ${e}${on ? ': selected' : ''}"></span>`;
  }
  return `<div class="experts" style="grid-template-columns:repeat(${cols},${size}px)">${cells}</div>
    <div class="note">${f.experts} routed experts${f.groups ? ` (one row per group: ${f.groups} groups × ${perGroup})` : ''}; ${f.topK} selected for one token (blue, illustrative)${f.groups ? `, all within ${f.topkGroups} groups` : ''}.${f.shared ? ` +${f.shared} shared expert always active.` : ''}</div>`;
}

function probBars() {
  const rest = 1 - NEXT.reduce((s, [, p]) => s + p, 0);
  return [...NEXT, ['(all other tokens)', rest]].map(([t, p], i) => `
    <div class="bar" title="${esc(t)}: p = ${p.toFixed(2)}">
      <span class="mono lbl">${i < NEXT.length ? `"${esc(t.trim())}"` : esc(t)}</span>
      <span class="track"><span style="width:${p * 100}%;background:${i < NEXT.length ? '#2563eb' : '#b9c0ca'}"></span></span>
      <span class="mono val">${p.toFixed(2)}</span>
    </div>`).join('');
}

// ---------- stages: structure + text for one model ----------
function buildStages(m) {
  const P = params(m), a = m.attention, f = m.ffn, H = m.hidden, L = m.layers, V = m.vocab;
  const tH = `${T} × ${fmt(H)}`;
  const kvB = kvValuesPerToken(m) * 2;

  const input = [
    { id: 'text', title: 'Text prompt', mono: true, viz: `"${esc(TOKENS.join(''))}"`,
      what: 'Input string. Every later stage operates on integer token IDs or floating-point tensors derived from it.',
      shapes: [['input', `"${TOKENS.join('')}"`]],
      links: [['tokenizer', 'input to']] },
    { id: 'tokenizer', title: 'Tokenizer', viz: TOKENS.map((t) => `<span class="chip">${show(t)}</span>`).join(''),
      what: `Splits the string into a sequence of T tokens with ${m.tokenizer}. BPE merges frequent byte sequences into single vocabulary entries; strings without a merged entry are encoded as several shorter tokens, down to single bytes. A leading space is part of the token (␣cat).`,
      shapes: [['input', 'string'], ['output', `T = ${T} tokens`], ['embedding rows V', fmt(V)]],
      note: 'V is the config vocab_size (embedding rows); it can exceed the tokenizer\'s entry count by padding rows.',
      links: [['token-ids', 'output to'], ['lm-head', 'same index space V as']] },
    { id: 'token-ids', title: 'Token IDs', mono: true, viz: `[${IDS.join(', ')}]`,
      what: `Each token is replaced by its integer index in the vocabulary, 0 ≤ id < ${fmt(V)}.`,
      shapes: [['output', `${T} integers`]], note: 'IDs shown are illustrative.',
      links: [['embedding', 'row indices for']] },
    { id: 'embedding', title: 'Token embedding', mono: true,
      viz: TOKENS.map((t, i) => `${show(t).padEnd(5)} → W_E[${IDS[i]}] = [${[0.12, -0.48, 0.91].map((v) => (v * (i % 2 ? -1 : 1) + i * 0.03).toFixed(2)).join(', ')}, … ${fmt(H)} values]`).join('\n'),
      what: `Lookup table W_E ∈ ℝ^(${fmt(V)} × ${fmt(H)}). Token ID i selects row i: a gather (indexing) operation, no matrix multiplication.${m.tied ? ' The same matrix is reused as the LM head (tied weights).' : ''}`,
      why: 'Output h₀ ∈ ℝ^(T × H) is the initial residual stream.',
      shapes: [['input', `${T} IDs`], ['output', tH], ['parameters', `${fmtM(P.embed)} (${pct(P.embed, P.total)} of total)`]],
      note: 'Vector values shown are illustrative.',
      links: [['token-ids', 'indexed by'], m.position === 'learned' ? ['pos-embed', 'summed with'] : ['residual', 'initializes'], ...(m.tied ? [['lm-head', 'shares weights with']] : [])] },
  ];
  if (m.position === 'learned') {
    input.push({ id: 'pos-embed', title: 'Position embedding (learned)', mono: true, viz: `h₀ = W_E[ids] + W_P[0 … ${T - 1}]`,
      what: `Learned table W_P ∈ ℝ^(${fmt(m.context)} × ${fmt(H)}). Row p is added to the token embedding at position p. The table size fixes the maximum sequence length at ${fmt(m.context)} tokens.`,
      shapes: [['output', tH], ['parameters', fmtM(P.pos)]],
      links: [['embedding', 'added to'], ['residual', 'initializes'], ['attention', 'provides the only position information to']] });
  }

  // attention
  let attention;
  if (a.kind === 'MLA') {
    const dqk = a.nopeDim + a.ropeDim, full = a.heads * (dqk + a.vDim);
    attention = { id: 'attention', title: 'Multi-head latent attention (MLA)', tag: 'mixes information across positions', viz: attentionGrid(),
      what: `Queries: c_Q = x·W_DQ ∈ ℝ^(T × ${a.qRank}), RMSNorm, then up-projection to ${a.heads} heads × ${dqk}. Keys and values: c_KV = x·W_DKV ∈ ℝ^(T × ${a.kvRank}), RMSNorm, then per-head up-projections K_C = c_KV·W_UK (${a.nopeDim}) and V = c_KV·W_UV (${a.vDim}). A separate key k_R = RoPE(x·W_KR) ∈ ℝ^(T × ${a.ropeDim}) is shared by all heads and concatenated to each head's K_C. Per head: S = Q·Kᵀ / √${dqk}, causal mask (S[i, j] = −∞ for j > i), A = softmax(S) row-wise, O = A·V. Output projection W_O: ${a.heads}·${a.vDim} → ${fmt(H)}. RoPE (θ = ${fmt(m.ropeTheta)}) is applied only to the ${a.ropeDim}-dim parts of q and k.`,
      why: `Only c_KV and k_R are cached: ${a.kvRank} + ${a.ropeDim} = ${a.kvRank + a.ropeDim} values per token per layer, versus ${a.heads}·(${dqk} + ${a.vDim}) = ${fmt(full)} for per-head K and V (${(full / (a.kvRank + a.ropeDim)).toFixed(0)}× smaller). At decode time W_UK can be folded into the query projection and W_UV into W_O, so per-head K and V are never materialized.`,
      shapes: [['input', tH], ['c_KV, k_R (cached)', `${T} × ${a.kvRank}, ${T} × ${a.ropeDim}`], ['Q', `${T} × ${a.heads} × ${dqk}`], ['K, V', `${T} × ${a.heads} × ${dqk}, ${T} × ${a.heads} × ${a.vDim}`], ['S, A', `${a.heads} × ${T} × ${T}  (O(T²))`], ['output', tH], ['parameters / layer', fmtM(P.attnLayer)]],
      links: [['norm1', 'input from'], ['add1', 'output to'], ['softmax', 'same softmax function as'], ['sampling', 'c_KV, k_R stored in the KV cache during']] };
  } else {
    const g = a.heads / kvHeads(m);
    const bias = a.bias === 'all' ? ' (all four projections have bias vectors)' : a.bias === 'qkv' ? ' (Q, K, V projections have bias vectors; W_O does not)' : '';
    attention = { id: 'attention', title: a.kind === 'GQA' ? 'Grouped-query attention (GQA)' : 'Multi-head attention (MHA)', tag: 'mixes information across positions', viz: attentionGrid(),
      what: `Q = x·W_Q, K = x·W_K, V = x·W_V${bias}. Q is split into ${a.heads} heads and K, V into ${kvHeads(m)} heads, each d_h = ${a.headDim}. Per head: S = Q·Kᵀ / √${a.headDim}, causal mask (S[i, j] = −∞ for j > i), A = softmax(S) row-wise, O = A·V. Head outputs are concatenated (${a.heads}·${a.headDim} = ${fmt(a.heads * a.headDim)}) and multiplied by W_O. ` +
        (m.position === 'rope' ? `RoPE (θ = ${fmt(m.ropeTheta)}) rotates each 2-dim pair of q and k by an angle proportional to the position before Q·Kᵀ, so each score depends on the offset i − j.` : 'No position operation here: absolute positions were added at the embedding.'),
      why: (a.kind === 'GQA' ? `Each K/V head is shared by ${g} query heads, so the K/V projections and the KV cache are ${g}× smaller than with ${a.heads} K/V heads. ` : 'One K/V head per query head. ') +
        'Attention is the only operation in a layer that combines values from different sequence positions; all other operations are applied per position.',
      shapes: [['input', tH], ['Q', `${T} × ${a.heads} × ${a.headDim}`], ['K, V', `${T} × ${kvHeads(m)} × ${a.headDim} each`], ['S, A', `${a.heads} × ${T} × ${T}  (O(T²))`], ['output', tH], ['parameters / layer', fmtM(P.attnLayer)]],
      links: [['norm1', 'input from'], ['add1', 'output to'], ['softmax', 'same softmax function as'], ['sampling', 'K, V stored in the KV cache during'], ...(m.position === 'learned' ? [['pos-embed', 'position information from']] : [])] };
  }

  const normText = m.norm === 'RMSNorm'
    ? `y = x / √(mean(x²) + ε) ⊙ g, ε = ${m.normEps}, g ∈ ℝ^${fmt(H)} learned. Computed per token over the H dimension.`
    : `y = (x − μ) / √(σ² + ε) ⊙ g + b, ε = ${m.normEps}, g, b ∈ ℝ^${fmt(H)} learned. μ, σ² are computed per token over the H dimension.`;

  const activation = { id: 'activation', title: `Activation: ${actName(m)}`, tag: 'element-wise non-linearity', viz: activationChart(m),
    what: m.ffn.act === 'silu'
      ? `SiLU(x) = x·σ(x), σ(x) = 1 / (1 + e^(−x)). Applied element-wise to the gate projection x·W_gate.`
      : `GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))), the tanh approximation. Applied element-wise to x·W_1 + b_1.`,
    why: m.ffn.gated
      ? 'The gate SiLU(x·W_gate) multiplies the up projection element-wise, scaling each intermediate channel by a non-linear function of x.'
      : 'Without the activation, W_1 and W_2 compose into a single H × H linear map.',
    shapes: [['input / output', `${T} × ${fmt(f.kind === 'moe' ? f.expertInter : f.inter)} (per FFN)`]],
    links: [[f.kind === 'moe' ? 'experts' : 'mlp', 'inside']] };

  const ffnFormula = m.ffn.gated
    ? `y = (${actName(m)}(x·W_gate) ⊙ (x·W_up))·W_down`
    : `y = ${actName(m)}(x·W_1 + b_1)·W_2 + b_2`;

  let ffn;
  if (f.kind === 'dense') {
    ffn = { id: 'mlp', title: `Feed-forward network (${ffnName(m)})`, tag: 'applied per position', row: true,
      pre: `${fmt(H)} → ${fmt(f.inter)}${f.gated ? ' (gate, up)' : ''}`, post: `→ ${fmt(H)}`, children: [activation],
      what: `${ffnFormula}, intermediate size F = ${fmt(f.inter)}. ${f.gated ? `W_gate, W_up ∈ ℝ^(${fmt(H)} × ${fmt(f.inter)}), W_down ∈ ℝ^(${fmt(f.inter)} × ${fmt(H)}).` : `W_1 ∈ ℝ^(${fmt(H)} × ${fmt(f.inter)}), W_2 ∈ ℝ^(${fmt(f.inter)} × ${fmt(H)}).`} Applied independently at each position.`,
      why: `${fmtM(P.ffnLayer)} parameters per layer; FFN weights are ${pct(P.ffnTotal, P.total)} of all parameters.`,
      shapes: [['input', tH], ['intermediate', `${T} × ${fmt(f.inter)}${f.gated ? ' (×2: gate, up)' : ''}`], ['output', tH], ['parameters / layer', fmtM(P.ffnLayer)]],
      links: [['norm2', 'input from'], ['activation', 'contains'], ['add2', 'output to']] };
  } else {
    const router = f.router === 'softmax'
      ? `Router logits r = x·W_r ∈ ℝ^${f.experts} (W_r: ${fmt(H)} × ${f.experts}). The top-${f.topK} logits are selected and normalized with softmax over the selected entries; the results are the gate weights g.`
      : `Affinity s = sigmoid(x·W_r) ∈ ℝ^${f.experts}. Selection ranks s + b, where b is a per-expert bias updated during training for load balancing; b is not used in the gate values. Group-limited: the ${f.experts} experts form ${f.groups} groups, the top ${f.topkGroups} groups are selected first, then the top ${f.topK} experts within them. Gates g = s of the selected experts, normalized to sum 1, multiplied by ${f.routedScale}.`;
    const expertsBox = { id: 'experts', title: `Expert FFN (${ffnName(m)})`, tag: `${f.topK} of ${f.experts} run per token`, inner: true, row: true,
      pre: `${fmt(H)} → ${fmt(f.expertInter)}`, post: `→ ${fmt(H)}`, children: [activation],
      what: `Each routed expert is an independent ${ffnName(m)}: ${ffnFormula}, intermediate size F_e = ${fmt(f.expertInter)}. Tokens are permuted into per-expert batches, processed with grouped matrix multiplications, and permuted back.${f.shared ? ` The ${f.shared} shared expert has the same shape and processes every token.` : ''}`,
      shapes: [['per expert', `tokens × ${fmt(H)} → ${fmt(f.expertInter)} → ${fmt(H)}`], ['parameters / expert', fmtM(P.perExpert)]],
      links: [['router', 'tokens assigned by'], ['activation', 'contains']] };
    const routerBox = { id: 'router', title: 'Router', tag: `top-${f.topK}`, inner: true, viz: routerGrid(m),
      what: router, shapes: [['input', tH], ['scores', `${T} × ${f.experts}`], ['selected', `${T} × ${f.topK} expert IDs + gate weights`]],
      note: 'Selected experts shown are illustrative.',
      links: [['experts', 'assigns tokens to']] };
    ffn = { id: 'moe', title: 'Mixture-of-experts FFN', tag: 'applied per position', children: [routerBox, expertsBox],
      what: `${f.experts} routed experts${f.shared ? ` and ${f.shared} shared expert` : ''} per layer. For each token: y = ${f.shared ? 'Shared(x) + ' : ''}Σ_{i ∈ top-${f.topK}} g_i · Expert_i(x).` +
        (f.denseLayers ? ` Layers 1–${f.denseLayers} use a dense ${ffnName(m)} FFN with F = ${fmt(f.denseInter)} instead; layers ${f.denseLayers + 1}–${L} use this block.` : ''),
      why: `Parameters used per token: ${fmtM(P.active)} of ${fmtM(P.total)} total (${pct(P.active, P.total)}). FLOPs per token scale with active parameters; weight memory scales with total parameters.`,
      shapes: [['input', tH], ['output', tH], ['parameters / MoE layer', fmtM(P.ffnLayer)]],
      links: [['norm2', 'input from'], ['router', 'contains'], ['experts', 'contains'], ['add2', 'output to']] };
  }

  const layer = [
    { id: 'norm1', title: `${normName(m)}`, size: 'small', what: normText, why: 'Pre-norm: applied to the sub-layer input only; the residual stream h is not modified.',
      shapes: [['input / output', tH]], links: [['residual', 'reads'], ['attention', 'output to']] },
    attention,
    { id: 'add1', title: '⊕  h ← h + Attention(norm(h))', size: 'small', add: true,
      what: 'Residual connection: the attention output is added element-wise to the residual stream.', shapes: [['output', tH]],
      links: [['residual', 'writes to'], ['norm2', 'output to']] },
    { id: 'norm2', title: `${normName(m)}`, size: 'small', what: normText, why: 'Separate learned g (and b) from the first norm.',
      shapes: [['input / output', tH]], links: [['residual', 'reads'], [ffn.id, 'output to']] },
    ffn,
    { id: 'add2', title: '⊕  h ← h + FFN(norm(h))', size: 'small', add: true,
      what: `Residual connection: the FFN output is added to the residual stream. This completes one layer; the result is the input to the next of ${L} layers, each with its own weights.`,
      shapes: [['output', tH]], links: [['residual', 'writes to'], ['norm1', 'next layer input to'], ['final-norm', 'after layer L, output to']] },
  ];

  const residual = { id: 'residual', title: 'Residual stream h',
    what: `h ∈ ℝ^(T × H), T = ${T}, H = ${fmt(H)}. Initialized by the embedding, updated additively by both sub-layers of all ${L} layers: h ← h + f(norm(h)). Its width H is constant through the network.`,
    why: 'The identity term gives every sub-layer output a direct additive path to the LM head and keeps gradient magnitudes stable through depth during training.',
    shapes: [['size', `${tH} at every layer`]],
    links: [[m.position === 'learned' ? 'pos-embed' : 'embedding', 'initialized by'], ['add1', 'updated by'], ['add2', 'updated by'], ['final-norm', 'read by']] };

  const output = [
    { id: 'final-norm', title: `Final ${normName(m)}`, size: 'small', what: `Normalization of the residual stream after layer ${L}. ${normText}`,
      shapes: [['input / output', tH]], links: [['residual', 'reads'], ['lm-head', 'output to']] },
    { id: 'lm-head', title: 'LM head → logits', mono: true,
      viz: `z = [ ${NEXT.slice(0, 3).map(([t, p]) => `${(17.2 + Math.log(p / NEXT[0][1])).toFixed(1)} "${esc(t.trim())}"`).join(', ')}, … ${fmt(V)} values ]`,
      what: `z = norm(h_T)·W_U, W_U ∈ ℝ^(${fmt(H)} × ${fmt(V)}): one logit per vocabulary entry. Only the last position T is needed to predict token T + 1.${m.tied ? ' Tied weights: W_U = W_Eᵀ, no additional parameters.' : ''}`,
      shapes: [['input', `1 × ${fmt(H)} (last position)`], ['output', `${fmt(V)} logits`], ['parameters', m.tied ? '0 (tied to W_E)' : fmtM(P.head)]],
      note: 'Logit values shown are illustrative.',
      links: [['final-norm', 'input from'], ['tokenizer', 'indexes the same vocabulary as'], ['softmax', 'output to'], ...(m.tied ? [['embedding', 'shares weights with']] : [])] },
    { id: 'softmax', title: 'Softmax → probabilities', viz: probBars(),
      what: `p_i = exp(z_i / τ) / Σ_j exp(z_j / τ) over all ${fmt(V)} entries; τ is the sampling temperature (τ = 1 leaves z unchanged). Result: p_i > 0 and Σ p_i = 1.`,
      why: 'Lower τ increases the ratio between the largest and the smaller probabilities; higher τ decreases it.',
      shapes: [['output', `${fmt(V)} probabilities`]], note: 'Probabilities shown are illustrative.',
      links: [['lm-head', 'input from'], ['attention', 'same function as in'], ['sampling', 'output to']] },
    { id: 'sampling', title: 'Next-token selection', mono: true, viz: `"${esc(TOKENS.join(''))}<b>${esc(NEXT[0][0])}</b>"`,
      what: 'Greedy decoding selects argmax_i p_i. Sampling draws i ~ p, optionally restricted to the k largest probabilities (top-k) or to the smallest set with cumulative probability ≥ p (top-p). The selected ID is appended to the sequence and the forward pass runs again for position T + 1 (autoregressive decoding).',
      why: `K and V of earlier positions do not change, so they are stored in the KV cache; each decode step computes Q, K, V only for the new position. KV cache size: ${fmt(kvValuesPerToken(m))} values = ${fmtBytes(kvB)} per token in BF16.`,
      shapes: [['output', '1 token ID, then repeat']],
      links: [['softmax', 'input from'], ['token-ids', 'appends to'], ['attention', 'KV cache stores values from']] },
  ];

  return { input, layer, residual, output };
}

// ---------- rendering ----------
let model, stages, byId, order, current = 0;

function boxHtml(s) {
  const cls = ['box', s.size || '', s.inner ? 'inner' : '', s.add ? 'add' : ''].filter(Boolean).join(' ');
  let body = s.viz ? `<div class="viz${s.mono ? ' mono pre' : ''}">${s.viz}</div>` : '';
  if (s.children) {
    const kids = s.children.map(boxHtml).join(s.row ? '' : '<div class="arrow">↓</div>');
    body += s.row
      ? `<div class="row"><div class="mini mono">${esc(s.pre)}</div>${kids}<div class="mini mono">${esc(s.post)}</div></div>`
      : `<div class="kids">${kids}</div>`;
  }
  return `<div class="${cls}" data-id="${s.id}"><div class="title">${esc(s.title)}${s.tag ? ` <span class="tag">${esc(s.tag)}</span>` : ''}</div>${body}</div>`;
}
const column = (list) => list.map(boxHtml).join('<div class="arrow">↓</div>');

function render(m) {
  model = m;
  stages = buildStages(m);
  byId = {};
  order = [];
  const walk = (s) => { byId[s.id] = s; order.push(s.id); (s.children || []).forEach(walk); };
  stages.input.forEach(walk);
  walk(stages.residual);
  stages.layer.forEach(walk);
  stages.output.forEach(walk);

  const P = params(m), f = m.ffn;
  const layerNote = f.kind === 'moe' && f.denseLayers
    ? `<p class="group-note">Layers 1–${f.denseLayers}: dense FFN (F = ${fmt(f.denseInter)}). Layers ${f.denseLayers + 1}–${m.layers}: MoE FFN, shown below.</p>` : '';
  document.getElementById('diagram').innerHTML = `
    <section class="group"><h2>1 · Input</h2>${column(stages.input)}</section>
    <div class="arrow">↓</div>
    <section class="group block"><h2>2 · Transformer layer <span class="repeat">× ${m.layers}</span></h2>${layerNote}
      <div class="block-body">
        <div class="residual" data-id="residual"><span class="title">residual stream h</span></div>
        <div class="block-steps">${column(stages.layer)}</div>
      </div></section>
    <div class="arrow">↓</div>
    <section class="group"><h2>3 · Output</h2>${column(stages.output)}
      <div class="loop">↺ the selected token is appended and the forward pass repeats for the next position</div></section>`;

  const active = f.kind === 'moe' ? `, ${fmtM(P.active)} active per token` : '';
  document.getElementById('model-line').innerHTML =
    `${esc(m.name)} (${esc(m.org)}, ${m.year}): L = ${m.layers}, H = ${fmt(m.hidden)}, V = ${fmt(m.vocab)}, context ${fmt(m.context)}, ` +
    `${fmtM(P.total)} parameters${active}. Source: <a href="${esc(m.source)}" target="_blank" rel="noreferrer">config.json</a>. ` +
    'Example values (token IDs, attention weights, expert selection, probabilities) are illustrative.';

  document.querySelectorAll('#models button').forEach((b) => b.classList.toggle('on', b.dataset.model === m.id));
}

function modelBar() {
  document.getElementById('models').innerHTML = window.MODELS.map((m) => {
    const P = params(m), f = m.ffn;
    const ffn = f.kind === 'moe' ? `MoE ${f.topK}/${f.experts}` : `dense ${ffnName(m)}`;
    const size = f.kind === 'moe' ? `${fmtM(P.total)} (${fmtM(P.active)} active)` : fmtM(P.total);
    return `<button data-model="${m.id}"><b>${esc(m.name)}</b><span>${size} · ${m.attention.kind} · ${ffn}</span></button>`;
  }).join('');
}

// ---------- selection: current, next, related ----------
function select(id, scroll = true) {
  const s = byId[id];
  if (!s) return;
  current = order.indexOf(id);
  const nextId = current < order.length - 1 ? order[current + 1] : null;
  const related = new Set(s.links.map(([to]) => to));

  document.querySelectorAll('.badge').forEach((b) => b.remove());
  document.querySelectorAll('[data-id]').forEach((el) => {
    const eid = el.dataset.id;
    el.classList.toggle('current', eid === id);
    el.classList.toggle('next', eid === nextId);
    el.classList.toggle('related', related.has(eid) && eid !== id && eid !== nextId);
  });
  const mark = (eid, text) => {
    const t = document.querySelector(`[data-id="${eid}"] > .title`);
    if (t) t.insertAdjacentHTML('beforeend', `<span class="badge ${text.toLowerCase()}">${text}</span>`);
  };
  mark(id, 'CURRENT');
  if (nextId) mark(nextId, 'NEXT');

  const rows = s.shapes.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join('');
  const links = s.links.filter(([to]) => byId[to]).map(([to, rel]) => `<li>${esc(rel)} <a data-go="${to}">${esc(byId[to].title)}</a></li>`).join('');
  document.getElementById('panel').innerHTML = `
    <div class="panel-current">
      <div class="panel-kicker">CURRENT · step ${current + 1} of ${order.length}</div>
      <h3>${esc(s.title)}</h3>
      <p>${esc(s.what)}</p>
      ${s.why ? `<h4>Detail</h4><p>${esc(s.why)}</p>` : ''}
      <h4>Shapes (${esc(model.name)}, T = ${T})</h4>
      <table>${rows}</table>
      ${s.note ? `<p class="note">${esc(s.note)}</p>` : ''}
      <h4>Connections</h4>
      <ul>${links}</ul>
    </div>
    ${nextId ? `<button class="panel-next" data-go="${nextId}"><span class="panel-kicker">NEXT</span>${esc(byId[nextId].title)} ▶</button>` : ''}`;
  document.getElementById('step-label').textContent = `Step ${current + 1} of ${order.length}: ${s.title}`;
  if (scroll) document.querySelector(`[data-id="${id}"]`).scrollIntoView({ block: 'center', behavior: 'smooth' });
  location.hash = `model=${model.id}&stage=${id}`;
}

function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  return { modelId: q.get('model'), stage: q.get('stage') };
}

document.addEventListener('click', (e) => {
  const mb = e.target.closest('#models button');
  if (mb) {
    const keep = order[current];
    render(window.MODELS.find((m) => m.id === mb.dataset.model));
    return select(byId[keep] ? keep : 'text', false);
  }
  const go = e.target.closest('[data-go]');
  if (go) return select(go.dataset.go);
  const box = e.target.closest('[data-id]');
  if (box) select(box.dataset.id, false);
});
document.getElementById('prev').onclick = () => select(order[(current - 1 + order.length) % order.length]);
document.getElementById('next').onclick = () => select(order[(current + 1) % order.length]);
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') document.getElementById('next').click();
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
});

modelBar();
const start = readHash();
render(window.MODELS.find((m) => m.id === start.modelId) || window.MODELS.find((m) => m.id === 'llama-3.1-8b') || window.MODELS[0]);
select(byId[start.stage] ? start.stage : 'text', false);
