// UI: model × phase × lens. Builds every view from models.js, hardware.js and calc.js.
// Text branches on architecture features and the scenario, never on a model name.
// The running example (token IDs, attention weights, expert selection, probabilities) is ILLUSTRATIVE.

const TOKENS = ['The', ' cat', ' sat', ' on', ' the'];
const IDS = [791, 8415, 7731, 389, 279];
const NEXT = [[' mat', 0.41], [' floor', 0.17], [' couch', 0.08], [' bed', 0.06], [' sofa', 0.05]];

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const show = (t) => esc(t.replace(/^ /, '␣'));
const $ = (id) => document.getElementById(id);
const pct = (a, b) => `${((100 * a) / b).toFixed(0)}%`;
const actName = (m) => (m.ffn.act === 'silu' ? 'SiLU' : 'GELU');
const ffnName = (m) => (m.ffn.gated ? (m.ffn.act === 'silu' ? 'SwiGLU' : 'GeGLU') : `${actName(m)} MLP`);
const techLink = (id) => { const t = window.TECHNIQUES.find((x) => x.id === id); return t ? `<a class="tech" href="technique.html?id=${id}">${esc(t.title)} →</a>` : ''; };

// ---------------- state ----------------
const state = { model: 'llama-3.1-8b', phase: 'inference', step: 'decode', lens: 'dataflow', stage: 'text', S: null };
const isTemplate = () => !window.MODELS.some((m) => m.id === state.model);
const currentModel = () => window.MODELS.find((m) => m.id === state.model);

function applyPreset() {
  const m = currentModel();
  if (!m) return;
  const p = m.presets[state.phase];
  const base = { phase: state.phase, gpu: window.GPUS[0].id, dp: 1, tp: 1, pp: 1, ep: 1, cp: 1, zero: 1, recompute: 'none', seq: 8192, mbs: 1, gbs: 64,
    batch: 16, prompt: 2048, output: 512, wdtype: 'bf16' };
  state.S = { ...base, ...p, phase: state.phase };
}

const HASH_KEYS = ['model', 'phase', 'step', 'lens', 'stage'];
const S_KEYS = ['gpu', 'dp', 'tp', 'pp', 'ep', 'cp', 'zero', 'recompute', 'seq', 'mbs', 'gbs', 'batch', 'prompt', 'output', 'wdtype'];
function writeHash() {
  const q = new URLSearchParams();
  HASH_KEYS.forEach((k) => q.set(k, state[k]));
  if (state.S) S_KEYS.forEach((k) => q.set(k, state.S[k]));
  history.replaceState(null, '', `#${q}`);
}
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  HASH_KEYS.forEach((k) => { if (q.get(k)) state[k] = q.get(k); });
  applyPreset();
  if (state.S) S_KEYS.forEach((k) => {
    const v = q.get(k);
    if (v !== null) state.S[k] = /^\d+$/.test(v) ? Number(v) : v;
  });
  // Links saved before the GPU list changed may name a GPU that no longer exists.
  if (state.S && !window.GPUS.some((g) => g.id === state.S.gpu)) state.S.gpu = window.GPUS[0].id;
}

// ---------------- top: model bar ----------------
function modelBar() {
  const llm = window.MODELS.map((m) => {
    const P = params(m), f = m.ffn;
    const size = f.kind === 'moe' ? `${fmtParams(P.total)} (${fmtParams(P.active)} active)` : fmtParams(P.total);
    return `<button data-model="${m.id}" title="${esc(m.why)}"><b>${esc(m.name)}</b><span>${size} · ${m.attention.kind} · ${f.kind === 'moe' ? `MoE ${f.topK}/${f.experts}` : 'dense'}</span></button>`;
  }).join('');
  const templates = window.FAMILIES.filter((f) => f.status === 'template')
    .map((f) => `<button data-model="${f.id}" class="tmpl"><b>${esc(f.title)}</b><span>template</span></button>`).join('');
  $('models').innerHTML = `<div class="fam"><span class="fam-title">Decoder-only LLM</span>${llm}</div>
    <div class="fam"><span class="fam-title">Vision / multimodal</span>${templates}</div>`;
}

// ---------------- controls ----------------
const OPTS = {
  dp: [1, 2, 4, 8, 16, 32, 64, 128, 256], tp: [1, 2, 4, 8, 16], pp: [1, 2, 4, 8, 16], ep: [1, 2, 4, 8, 16, 32, 64, 128, 256], cp: [1, 2, 4, 8, 16],
  zero: [0, 1, 2, 3], recompute: ['none', 'full'], seq: [2048, 4096, 8192, 16384, 32768, 131072], mbs: [1, 2, 4, 8], wdtype: ['bf16', 'fp8'],
};
const LABEL = { dp: 'DP', tp: 'TP', pp: 'PP', ep: 'EP', cp: 'CP', zero: 'ZeRO', recompute: 'recompute', seq: 'seq', mbs: 'micro-batch', gbs: 'global batch (seq)',
  batch: 'batch (seq)', prompt: 'prompt', output: 'output', wdtype: 'weights', gpu: 'GPU' };

function controls() {
  const tmpl = isTemplate(), S = state.S;
  const btn = (key, val, label) => `<button class="${state[key] === val ? 'on' : ''}" data-set="${key}" data-val="${val}">${label}</button>`;
  let html = `<div class="seg">${btn('phase', 'training', 'Training')}${btn('phase', 'inference', 'Inference')}</div>`;
  if (state.phase === 'inference' && !tmpl) html += `<div class="seg">${btn('step', 'prefill', 'Prefill')}${btn('step', 'decode', 'Decode')}</div>`;
  html += `<div class="seg lens">${btn('lens', 'dataflow', 'Data flow')}${btn('lens', 'memory', 'Memory')}${btn('lens', 'parallel', 'Parallelism')}</div>`;
  if (!tmpl) {
    const sel = (k, opts) => {
      const list = opts.includes(S[k]) ? opts : [...opts, S[k]].sort((a, b) => (typeof a === 'number' ? a - b : 0));
      return `<label>${LABEL[k]} <select data-s="${k}">${list.map((o) => `<option ${o === S[k] ? 'selected' : ''}>${o}</option>`).join('')}</select></label>`;
    };
    const num = (k) => `<label>${LABEL[k]} <input type="number" min="1" data-s="${k}" value="${S[k]}"></label>`;
    // Fixed while hardware.js lists one GPU; becomes a choice when entries are added there.
    const gpu = `<label title="${esc(gpuOf(S).assumed || '')}">GPU <select data-s="gpu" ${window.GPUS.length === 1 ? 'disabled' : ''}>${window.GPUS.map((g) => `<option value="${g.id}" ${g.id === S.gpu ? 'selected' : ''}>${esc(g.name)}</option>`).join('')}</select></label>`;
    const common = [gpu, sel('dp', OPTS.dp), sel('tp', OPTS.tp), sel('pp', OPTS.pp), sel('ep', OPTS.ep), sel('cp', OPTS.cp)];
    const phaseOnly = state.phase === 'training'
      ? [sel('zero', OPTS.zero), sel('recompute', OPTS.recompute), sel('seq', OPTS.seq), sel('mbs', OPTS.mbs), num('gbs')]
      : [sel('wdtype', OPTS.wdtype), num('batch'), num('prompt'), num('output')];
    html += `<div class="scenario">${[...common, ...phaseOnly].join('')}
      <span class="gpus">= ${fmtN(nGPUs(S))} GPU${nGPUs(S) > 1 ? 's' : ''} (${fmtN(layout(currentModel(), S).nodes)} node${layout(currentModel(), S).nodes > 1 ? 's' : ''} of ${gpuOf(S).perNode})</span>
      <button data-preset>Reset to preset</button></div>`;
    const m = currentModel();
    const issues = validate(m, S);
    html += `<p class="preset-note">Preset: ${esc(m.presets[state.phase].note)}</p>`;
    if (issues.length) html += `<p class="issues">${issues.map(esc).join(' · ')}</p>`;
  }
  $('controls').innerHTML = html;
}

// Target system, from hardware.js (source: HPC_spec_v1.md).
function systemLine() {
  const sys = window.SYSTEM, G = window.GPUS[0];
  if (!sys) return '';
  return `System: <b>${esc(sys.name)}</b> (${esc(sys.platform)}): ${G.perNode} × ${esc(G.name)} per node (${fmtBytes(G.hbm)} usable HBM each), ${esc(G.nvlinkName)} inside a node, ${esc(G.nicName)} between nodes; ${sys.gpus} GPUs in ${sys.gpuNodes} nodes as of ${esc(sys.asOf)}. Source: <a href="${esc(sys.source)}">${esc(sys.source)}</a>.`;
}

function modelLine() {
  if (isTemplate()) {
    const f = window.FAMILIES.find((x) => x.id === state.model);
    $('model-line').innerHTML = `<b>${esc(f.title)}</b>: template only. Stage outline is shown; configs, numbers and explanations are not written yet.`;
    return;
  }
  const m = currentModel(), P = params(m);
  $('model-line').innerHTML = `<b>${esc(m.name)}</b> (${esc(m.org)}, ${m.year}). ${esc(m.why)}<br>
    L = ${m.layers}, H = ${fmtN(m.hidden)}, V = ${fmtN(m.vocab)}, context ${fmtN(m.context)}, ${fmtParams(P.total)} parameters${m.ffn.kind === 'moe' ? `, ${fmtParams(P.active)} active per token` : ''}.
    Source: <a href="${esc(m.source)}" target="_blank" rel="noreferrer">config.json</a>${m.sourceNote ? ` (${esc(m.sourceNote)})` : ''}.<br>${systemLine()}`;
}

// ---------------- small visuals (illustrative example) ----------------
function attentionGrid() {
  const T = TOKENS.length;
  const raw = [[0, 0, 0, 0, 0], [1.0, 1.5, 0, 0, 0], [0.3, 2.2, 1.0, 0, 0], [0.2, 1.0, 1.8, 1.2, 0], [0.1, 2.0, 0.8, 1.6, 0.9]];
  const w = raw.map((r, i) => { const e = r.map((s, j) => (j <= i ? Math.exp(s) : 0)); const z = e.reduce((a, b) => a + b, 0); return e.map((x) => x / z); });
  const c = 30, left = 40, top = 26;
  let svg = `<svg width="${left + c * T + 4}" height="${top + c * T + 4}" role="img" aria-label="attention weights">`;
  TOKENS.forEach((t, j) => { svg += `<text x="${left + j * c + c / 2}" y="${top - 6}" font-size="10" text-anchor="middle" fill="#4b5563">${show(t)}</text>`; });
  TOKENS.forEach((t, i) => {
    svg += `<text x="${left - 5}" y="${top + i * c + c / 2 + 4}" font-size="10" text-anchor="end" fill="#4b5563">${show(t)}</text>`;
    w[i].forEach((v, j) => {
      const masked = j > i;
      svg += `<rect x="${left + j * c + 1}" y="${top + i * c + 1}" width="${c - 2}" height="${c - 2}" rx="2" fill="${masked ? '#eef0f3' : `rgba(37,99,235,${0.08 + 0.92 * v})`}"><title>${masked ? `A[${i},${j}] = 0 (causal mask)` : `A[${i},${j}] = ${v.toFixed(2)}`}</title></rect>`;
    });
  });
  return `<div class="viz-row">${svg}<div class="note">A = softmax(QKᵀ/√d) for one head, example T = 5.<br>Row i: query position i. Grey: masked (j &gt; i).<br>Values illustrative.</div></div>`;
}

function activationChart(m) {
  const W = 200, H = 90, x0 = -5, x1 = 4, y0 = -1, y1 = 4;
  const sx = (x) => ((x - x0) / (x1 - x0)) * W, sy = (y) => H - ((y - y0) / (y1 - y0)) * H;
  const fn = m.ffn.act === 'silu' ? (x) => x / (1 + Math.exp(-x)) : (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  const pts = []; for (let x = x0; x <= x1 + 1e-9; x += 0.1) pts.push(`${sx(x).toFixed(1)},${sy(fn(x)).toFixed(1)}`);
  return `<div class="viz-row"><svg width="${W}" height="${H}" role="img" aria-label="${actName(m)}">
    <line x1="0" y1="${sy(0)}" x2="${W}" y2="${sy(0)}" stroke="#d5dae1"/><line x1="${sx(0)}" y1="0" x2="${sx(0)}" y2="${H}" stroke="#d5dae1"/>
    <line x1="${sx(-1)}" y1="${sy(-1)}" x2="${sx(4)}" y2="${sy(4)}" stroke="#b9c0ca" stroke-dasharray="3 3"/>
    <polyline points="${pts.join(' ')}" fill="none" stroke="#2563eb" stroke-width="2"/></svg>
    <div class="note">Solid: ${actName(m)}(x). Dashed: y = x.</div></div>`;
}

function routerGrid(m) {
  const f = m.ffn, size = f.experts > 32 ? 11 : 24, groups = f.groups || 1, per = f.experts / groups;
  const chosen = Array.from({ length: f.topkGroups || 1 }, (_, i) => (i * 3 + 1) % groups);
  const picked = new Set(Array.from({ length: f.topK }, (_, i) => chosen[i % chosen.length] * per + ((i * 7 + 3) % per)));
  let cells = '';
  for (let e = 0; e < f.experts; e++) cells += `<span class="ex${picked.has(e) ? ' on' : ''}" style="width:${size}px;height:${size}px" title="expert ${e}"></span>`;
  return `<div class="experts" style="grid-template-columns:repeat(${Math.min(per, 32)},${size}px)">${cells}</div>
    <div class="note">${f.experts} routed experts${f.groups ? ` (row = group, ${f.groups} × ${per})` : ''}; ${f.topK} selected for one token, illustrative${f.groups ? `, within ${f.topkGroups} groups` : ''}.${f.shared ? ` + ${f.shared} shared expert.` : ''}</div>`;
}

function probBars() {
  const rest = 1 - NEXT.reduce((s, [, p]) => s + p, 0);
  return [...NEXT, ['(all other tokens)', rest]].map(([t, p], i) => `<div class="bar" title="p = ${p.toFixed(2)}">
    <span class="mono lbl">${i < NEXT.length ? `"${esc(t.trim())}"` : esc(t)}</span>
    <span class="track"><span style="width:${p * 100}%;background:${i < NEXT.length ? '#2563eb' : '#b9c0ca'}"></span></span>
    <span class="mono val">${p.toFixed(2)}</span></div>`).join('') + '<div class="note">Probabilities illustrative.</div>';
}

// ---------------- data flow: stage list for one model + scenario ----------------
function shapeCtx(m, S) {
  const Ly = layout(m, S);
  if (S.phase === 'training') return { B: S.mbs, T: S.seq, Tkv: S.seq, tokGPU: S.mbs * S.seq / S.cp, label: `micro-batch B = ${S.mbs}, T = ${fmtN(S.seq)}` };
  if (state.step === 'prefill') return { B: Ly.bReplica, T: S.prompt, Tkv: S.prompt, tokGPU: Ly.bReplica * S.prompt / S.cp, label: `prefill, B = ${Ly.bReplica} per replica, T = ${fmtN(S.prompt)}` };
  const ctx = S.prompt + S.output;
  return { B: Ly.bReplica, T: 1, Tkv: ctx, tokGPU: Ly.bReplica, label: `decode, B = ${Ly.bReplica} per replica, T = 1 new position, KV length ${fmtN(ctx)}` };
}

function collectiveRows(m, S) {
  return S.phase === 'training' ? trainingCollectives(m, S) : inferenceCollectives(m, S, state.step);
}

function markerStage(id, title, what, rows, tech, links) {
  const r = rows[0];
  return { id, title, kind: 'coll', size: 'small', tag: r ? `${r.link} · ${fmtBytes(r.volume)} / GPU per ${S_unit()}` : '',
    what, coll: rows, tech, links };
}
const S_unit = () => (state.S.phase === 'training' ? 'step' : state.step === 'prefill' ? 'prefill' : 'decode step');

// 1F1B bubble: idle time relative to compute is (PP − 1)/m; as a share of the whole step it is (PP − 1)/(m + PP − 1).
function bubbleText(S, Ly) {
  const p = S.pp, m = Ly.micro;
  return `Bubble: (PP − 1)/m = ${p - 1}/${fmtN(m)} = ${(((p - 1) / m) * 100).toFixed(1)}% of compute time, i.e. ${(((p - 1) / (m + p - 1)) * 100).toFixed(1)}% of the step idle. Interleaved schedules with v virtual stages per GPU divide it by v.`;
}

function buildStages(m, S) {
  const P = params(m), Ly = layout(m, S), a = m.attention, f = m.ffn, H = m.hidden, L = m.layers, V = m.vocab;
  const d = attnDims(m), sc = shapeCtx(m, S), train = S.phase === 'training';
  const wb = train ? 2 : BYTES[S.wdtype], wdt = train ? 'BF16' : S.wdtype.toUpperCase();
  const BTH = `${sc.B} × ${fmtN(sc.T)} × ${fmtN(H)}`;
  const coll = collectiveRows(m, S);
  const byDim = (dim) => coll.filter((r) => r.dim === dim);
  const tokG = sc.tokGPU;

  const input = [
    { id: 'text', title: 'Text prompt', mono: true, viz: `"${esc(TOKENS.join(''))}" <span class="note">(example)</span>`,
      what: 'Input string. Every later stage operates on integer token IDs or floating-point tensors derived from it.',
      shapes: [['this scenario', sc.label]], links: [['tokenizer', 'input to']] },
    { id: 'tokenizer', title: 'Tokenizer', viz: TOKENS.map((t) => `<span class="chip">${show(t)}</span>`).join(''),
      what: `Splits the string into tokens with ${m.tokenizer}. BPE merges frequent byte sequences into single vocabulary entries; other strings are encoded as several shorter tokens, down to single bytes. Runs on the CPU.`,
      shapes: [['vocabulary rows V', fmtN(V)]], hw: [['tokenizer', 'CPU', '—', 'host-side; output is copied to the GPU as int32 IDs']],
      links: [['token-ids', 'output to'], ['lm-head', 'same index space V as']] },
    { id: 'token-ids', title: 'Token IDs', mono: true, viz: `[${IDS.join(', ')}] <span class="note">(illustrative)</span>`,
      what: `Integer index of each token in the vocabulary, 0 ≤ id < ${fmtN(V)}. Copied host → GPU over PCIe.`,
      shapes: [['IDs', `${sc.B} × ${fmtN(sc.T)} int32`]],
      hw: [['token IDs', 'host RAM → GPU HBM (PCIe)', fmtBytes(sc.B * sc.T * 4), 'negligible next to weights']],
      links: [['embedding', 'row indices for']] },
    { id: 'embedding', title: 'Token embedding', mono: true,
      viz: TOKENS.map((t, i) => `${show(t).padEnd(5)} → W_E[${IDS[i]}] ∈ ℝ^${fmtN(H)}`).join('\n'),
      what: `Lookup table W_E ∈ ℝ^(${fmtN(V)} × ${fmtN(H)}). Token ID i selects row i: a gather, no matrix multiplication.${m.tied ? ' W_E is also the LM head (tied).' : ''} Under TP the table is vocab-parallel: each rank holds V/TP rows and an all-reduce assembles the output.`,
      shapes: [['output h₀', BTH]],
      hw: [['W_E', 'HBM', fmtBytes(P.embed / S.tp * wb), `${wdt}, V/TP rows per GPU`], ['h₀', 'HBM', fmtBytes(tokG * H * 2), 'BF16, this GPU\'s tokens']],
      tech: ['mixed-precision'], links: [['token-ids', 'indexed by'], ['residual', 'initializes']] },
  ];
  if (m.position === 'learned') input.push({ id: 'pos-embed', title: 'Position embedding (learned)', what: `W_P ∈ ℝ^(${fmtN(m.context)} × ${fmtN(H)}); row p is added at position p.`, shapes: [['output', BTH]], links: [['embedding', 'added to']] });

  // attention
  const qkv = a.kind === 'MLA'
    ? `Queries: c_Q = x·W_DQ (${a.qRank}), RMSNorm, up-projection to ${a.heads} heads × ${d.dqk}. Keys/values: c_KV = x·W_DKV ∈ ℝ^(T × ${a.kvRank}), RMSNorm, per-head up-projections K_C = c_KV·W_UK (${a.nopeDim}) and V = c_KV·W_UV (${a.vDim}); a RoPE key k_R ∈ ℝ^${a.ropeDim} is shared by all heads and concatenated to K_C. Only c_KV and k_R (${a.kvRank + a.ropeDim} values per token per layer) are cached.`
    : `Q = x·W_Q, K = x·W_K, V = x·W_V${a.bias === 'qkv' ? ' (with Q/K/V bias)' : ''}; Q split into ${a.heads} heads, K and V into ${d.kv} heads of d = ${a.headDim}${a.kind === 'GQA' ? ` (each K/V head shared by ${a.heads / d.kv} query heads)` : ''}.`;
  const kvPerLayerGPU = kvBytesPerTokenGPU(m, S) / Ly.layersPerStage;
  const attnHw = [
    ['W_Q, W_K, W_V, W_O (one layer)', 'HBM', fmtBytes(P.attnLayer / S.tp * wb), `${wdt}; column-parallel Q/K/V, row-parallel W_O over TP ${S.tp}`],
    ['Q, K, V (this layer)', 'HBM', fmtBytes(tokG * ((d.heads / S.tp) * d.dqk + 2 * Math.ceil((d.kv ?? d.heads) / S.tp) * d.dqk) * 2), `BF16, ${fmtN(tokG)} tokens on this GPU`],
    ['score tiles S = Q·Kᵀ', 'SRAM (per SM)', `≤ ${fmtBytes(gpuOf(S).sramPerSM)} / SM`, 'FlashAttention: computed tile by tile on-chip, never written to HBM'],
  ];
  if (!train) attnHw.push(['KV cache (this layer)', 'HBM', fmtBytes(kvPerLayerGPU * Ly.bReplica * (S.prompt + S.output)), state.step === 'prefill' ? 'written for all prompt positions' : 'read in full, one position appended']);
  if (train) attnHw.push(['saved for backward', 'HBM', '—', 'Q, K, V, O and softmax log-sum-exp (unless recomputed)']);
  const attnChildren = train ? [] : [{ id: 'kv-cache', title: 'KV cache', inner: true, tag: state.step === 'prefill' ? 'write T positions' : 'read all, append 1',
    what: `Stores ${a.kind === 'MLA' ? 'c_KV and k_R' : 'K and V'} of every processed position: ${fmtN(kvPerTokenLayer(m))} values per token per layer, ${fmtBytes(kvPerTokenLayer(m) * L * 2)} per token for the whole model in BF16. Prefill writes all prompt positions; each decode step reads all cached positions and appends one. Size grows linearly with context length and batch.`,
    shapes: [['per GPU now', fmtBytes(kvBytesPerTokenGPU(m, S) * Ly.bReplica * (S.prompt + S.output))]],
    hw: [['KV cache', 'HBM', fmtBytes(kvBytesPerTokenGPU(m, S) * Ly.bReplica * (S.prompt + S.output)), m.attention.kind === 'MLA' ? 'latent replicated across TP ranks' : 'sharded by KV head across TP']],
    tech: ['kv-cache', 'paged-kv-cache', 'continuous-batching'], links: [['attention', 'read and written by'], ['sampling', 'grows with each token from']] }];
  const attention = { id: 'attention', title: { MHA: 'Multi-head attention', GQA: 'Grouped-query attention (GQA)', MLA: 'Multi-head latent attention (MLA)' }[a.kind],
    tag: 'mixes information across positions', viz: attentionGrid(), children: attnChildren,
    what: `${qkv} Per head: S = Q·Kᵀ/√${d.dqk}, causal mask (S[i, j] = −∞ for j > i), A = softmax(S), O = A·V; heads concatenated and multiplied by W_O. ${m.position === 'rope' ? `RoPE (θ = ${fmtN(m.ropeTheta)}) rotates q and k by position before Q·Kᵀ.` : ''}`,
    why: 'The only operation in a layer that combines values from different positions; all others are applied per position. Score computation is O(T·T_kv) per head.',
    shapes: [['input x', BTH], ['Q', `${sc.B} × ${fmtN(sc.T)} × ${d.heads} × ${d.dqk}`], ['K, V', `${sc.B} × ${fmtN(sc.Tkv)} × ${d.kv ?? d.heads} × ${d.dqk}`], ['S, A', `${sc.B} × ${d.heads} × ${fmtN(sc.T)} × ${fmtN(sc.Tkv)}`], ['params / layer', fmtParams(P.attnLayer)]],
    hw: attnHw, tech: ['flash-attention', a.kind === 'GQA' ? 'gqa' : a.kind === 'MLA' ? 'mla' : null, m.position === 'rope' ? 'rope' : null].filter(Boolean),
    links: [['norm1', 'input from'], ['add1', 'output to']] };

  const norm = (id) => ({ id, title: m.norm, size: 'small',
    what: m.norm === 'RMSNorm' ? `y = x / √(mean(x²) + ε) ⊙ g, ε = ${m.normEps}; per token over H.` : `y = (x − μ)/√(σ² + ε) ⊙ g + b; per token over H.`,
    why: 'Pre-norm: applied to the sub-layer input; the residual stream is not modified. With SP the norm runs on a 1/TP slice of the sequence.',
    shapes: [['in / out', BTH]], links: [['residual', 'reads']] });

  const activation = { id: 'activation', title: `Activation: ${actName(m)}`, tag: 'element-wise', viz: activationChart(m), inner: true,
    what: m.ffn.act === 'silu' ? 'SiLU(x) = x·σ(x), applied element-wise to the gate projection x·W_gate.' : 'GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))), applied element-wise.',
    why: m.ffn.gated ? 'SiLU(x·W_gate) multiplies x·W_up element-wise; without the non-linearity the FFN would be a bilinear map of x.' : 'Without it W_1·W_2 collapse into one linear map.',
    shapes: [['in / out', `${sc.B} × ${fmtN(sc.T)} × ${fmtN(f.kind === 'moe' ? f.expertInter : f.inter)}`]], tech: m.ffn.gated ? ['swiglu'] : [],
    links: [[f.kind === 'moe' ? 'experts' : 'mlp', 'inside']] };
  const ffnFormula = m.ffn.gated ? `y = (${actName(m)}(x·W_gate) ⊙ x·W_up)·W_down` : `y = ${actName(m)}(x·W_1)·W_2`;

  let ffn;
  if (f.kind === 'dense') {
    ffn = { id: 'mlp', title: `Feed-forward network (${ffnName(m)})`, tag: 'per position', row: true,
      pre: `${fmtN(H)} → ${fmtN(f.inter)}`, post: `→ ${fmtN(H)}`, children: [activation],
      what: `${ffnFormula}, F = ${fmtN(f.inter)}. Under TP, W_gate and W_up are column-parallel (each rank computes F/TP channels) and W_down is row-parallel (partial sums, then all-reduce).`,
      why: `${fmtParams(P.ffnDenseLayer)} parameters per layer; FFN weights are ${pct(P.ffnTotal, P.total)} of all parameters.`,
      shapes: [['input', BTH], ['intermediate', `${sc.B} × ${fmtN(sc.T)} × ${fmtN(f.inter)}`], ['params / layer', fmtParams(P.ffnDenseLayer)]],
      hw: [['W_gate, W_up, W_down (one layer)', 'HBM', fmtBytes(P.ffnDenseLayer / S.tp * wb), `${wdt}, split over TP ${S.tp}`],
        ['intermediate activations', 'HBM', fmtBytes(tokG * (f.inter / S.tp) * (f.gated ? 3 : 2) * 2), 'BF16, this GPU']],
      tech: ['swiglu', 'tensor-parallel'].filter((t) => t !== 'swiglu' || m.ffn.gated), links: [['norm2', 'input from'], ['activation', 'contains'], ['add2', 'output to']] };
  } else {
    const epRows = byDim('ep');
    const expertsHere = S.ep > 1 ? `${f.experts / S.ep} whole experts per GPU (EP ${S.ep})` : `all ${f.experts} experts, each split 1/TP ${S.tp}`;
    const routerBox = { id: 'router', title: 'Router', tag: `top-${f.topK}`, inner: true, viz: routerGrid(m),
      what: f.router === 'softmax'
        ? `r = x·W_r ∈ ℝ^${f.experts}; top-${f.topK} logits selected; gates g = softmax over the selected logits.`
        : `s = sigmoid(x·W_r) ∈ ℝ^${f.experts}; selection ranks s + b (per-expert load-balancing bias, not used in gates); group-limited: top ${f.topkGroups} of ${f.groups} groups, then top ${f.topK} experts; g = normalized s × ${f.routedScale}.`,
      shapes: [['scores', `${sc.B} × ${fmtN(sc.T)} × ${f.experts}`]], tech: ['moe'], links: [['experts', 'assigns tokens to']] };
    const dispatch = S.ep > 1 ? [markerStage('coll-ep-dispatch', `EP all-to-all: dispatch`, 'Each token\'s hidden vector is sent to the EP ranks holding its selected experts. The volume assumes each routed copy is sent separately in BF16; node-limited routing and FP8 dispatch reduce inter-node bytes.', epRows, ['expert-parallel', 'collectives'], [['experts', 'delivers tokens to']])] : [];
    const combine = S.ep > 1 ? [markerStage('coll-ep-combine', `EP all-to-all: combine`, 'Expert outputs are returned to each token\'s home rank and summed with the gate weights.', epRows, ['expert-parallel', 'collectives'], [['add2', 'output to']])] : [];
    const expertsBox = { id: 'experts', title: `Expert FFN (${ffnName(m)})`, tag: `${f.topK} of ${f.experts} per token`, inner: true, row: true,
      pre: `${fmtN(H)} → ${fmtN(f.expertInter)}`, post: `→ ${fmtN(H)}`, children: [activation],
      what: `Each expert: ${ffnFormula}, F_e = ${fmtN(f.expertInter)}. Tokens are grouped per expert and processed with grouped GEMMs.${f.shared ? ` ${f.shared} shared expert processes every token.` : ''}`,
      shapes: [['params / expert', fmtParams(P.perExpert)]],
      hw: [['routed expert weights (one layer)', 'HBM', fmtBytes((f.experts * P.perExpert) / Ly.g * wb), `${wdt}; ${expertsHere}`],
        ...(f.shared ? [['shared expert (one layer)', 'HBM', fmtBytes(f.shared * P.perExpert / S.tp * wb), 'split over TP']] : [])],
      tech: ['moe', 'expert-parallel'], links: [['router', 'tokens assigned by'], ['activation', 'contains']] };
    ffn = { id: 'moe', title: 'Mixture-of-experts FFN', tag: 'per position', children: [routerBox, ...dispatch, expertsBox, ...combine],
      what: `${f.experts} routed experts${f.shared ? ` + ${f.shared} shared` : ''}; y = ${f.shared ? 'Shared(x) + ' : ''}Σ_{top-${f.topK}} g_i·Expert_i(x).${f.denseLayers ? ` Layers 1–${f.denseLayers} use a dense FFN (F = ${fmtN(f.denseInter)}).` : ''}`,
      why: `Per token ${fmtParams(P.active)} of ${fmtParams(P.total)} parameters are used (${pct(P.active, P.total)}). FLOPs scale with active parameters; weight memory with total parameters.`,
      shapes: [['input', BTH], ['params / MoE layer', fmtParams(P.router + (f.experts + f.shared) * P.perExpert)]],
      hw: [['dispatch buffer', 'HBM', fmtBytes((tokG / S.tp) * f.topK * H * 2), 'BF16 copies of tokens routed from this GPU']],
      tech: ['moe', 'expert-parallel'], links: [['norm2', 'input from'], ['router', 'contains'], ['add2', 'output to']] };
  }

  const tpRows = byDim('tp');
  const tpMarker = (id, after) => (S.tp > 1 ? [markerStage(id, `TP all-reduce after ${after}`,
    `Row-parallel ${after} leaves partial sums on each of the ${S.tp} TP ranks; all-reduce sums them so every rank holds the full output. With sequence parallelism this is a reduce-scatter here and an all-gather before the next column-parallel projection (same bytes).`,
    tpRows, ['tensor-parallel', 'sequence-parallel', 'collectives'], [['residual', 'result added to']])] : []);
  const cpMarker = S.cp > 1 && (train || state.step === 'prefill') ? [markerStage('coll-cp', 'CP ring exchange of K, V',
    `The sequence is split into ${S.cp} chunks. Each rank computes attention for its query chunk while K, V chunks are passed around the ring (P2P), so every query block is scored against all earlier keys.`,
    byDim('cp'), ['context-parallel', 'collectives'], [['attention', 'feeds']])] : [];
  const z3 = train && S.zero === 3 && Ly.dpDense > 1 ? [markerStage('coll-zero3', 'ZeRO-3 all-gather of layer weights',
    `Weights are sharded across ${Ly.dpDense} DP ranks. Before each layer's forward and backward, its full weights are all-gathered and freed afterwards.`,
    byDim('dp').filter((r) => r.op === 'all-gather'), ['zero', 'fsdp', 'collectives'], [['norm1', 'precedes'], ['attention', 'supplies weights to']])] : [];
  const ppMarker = S.pp > 1 ? [markerStage('coll-pp', `PP stage boundary (every ${Ly.layersPerStage} layers)`,
    `Layers are split into ${S.pp} stages of ⌈${L}/${S.pp}⌉ = ${Ly.layersPerStage} layers. At each boundary the residual stream is sent to the next stage (P2P)${train ? '; backward sends the activation gradient back' : ''}.${train ? ` ${bubbleText(S, Ly)}` : ' Stages run one after another for a given batch, so PP does not reduce per-token latency.'}`,
    byDim('pp'), ['pipeline-parallel', 'collectives'], [['norm1', 'next stage starts at']])] : [];

  const layer = [...z3, norm('norm1'), ...cpMarker, attention, ...tpMarker('coll-tp-attn', 'W_O'),
    { id: 'add1', title: '⊕  h ← h + Attention(norm(h))', size: 'small', add: true, what: 'Residual add, element-wise.', shapes: [['output', BTH]], links: [['residual', 'writes to']] },
    norm('norm2'), ffn, ...(f.kind === 'dense' || S.ep === 1 ? tpMarker('coll-tp-ffn', f.kind === 'dense' ? 'W_down' : 'the experts') : []),
    { id: 'add2', title: '⊕  h ← h + FFN(norm(h))', size: 'small', add: true, what: `Residual add. Completes one layer; the next of ${L} layers starts.`, shapes: [['output', BTH]], links: [['residual', 'writes to']] },
    ...ppMarker];

  const residual = { id: 'residual', title: 'Residual stream h',
    what: `h ∈ ℝ^(B × T × H), H = ${fmtN(H)}; updated additively by both sub-layers of all ${L} layers. Lives in HBM; with TP it is replicated on every TP rank (or split along T with SP).`,
    shapes: [['size', BTH]], hw: [['h', 'HBM', fmtBytes(tokG * H * 2), 'BF16, this GPU (÷TP with SP)']],
    tech: ['sequence-parallel'], links: [['embedding', 'initialized by'], ['add1', 'updated by'], ['add2', 'updated by']] };

  const lmW = m.tied ? 0 : P.head / S.tp * wb;
  const output = [
    { id: 'final-norm', title: `Final ${m.norm}`, size: 'small', what: `Normalization after layer ${L}.`, shapes: [['in / out', BTH]], links: [['lm-head', 'output to']] },
    { id: 'lm-head', title: 'LM head → logits', mono: true, viz: `z = h·W_U ∈ ℝ^${fmtN(V)}`,
      what: `z = h·W_U, W_U ∈ ℝ^(${fmtN(H)} × ${fmtN(V)}); vocab-parallel under TP. ${train ? 'Training computes logits for all T positions.' : 'Inference needs only the last position.'}`,
      shapes: [['logits', train ? `${sc.B} × ${fmtN(sc.T)} × ${fmtN(V)}` : `${sc.B} × ${fmtN(V)}`]],
      hw: [['W_U', 'HBM', m.tied ? 'shared with W_E' : fmtBytes(lmW), `${wdt}, V/TP columns`], ['logits', 'HBM', fmtBytes((train ? tokG : sc.B) * V / S.tp * 4), 'FP32, this GPU']],
      links: [['softmax', 'output to'], ['tokenizer', 'same vocabulary as']] },
    { id: 'softmax', title: 'Softmax → probabilities', viz: probBars(),
      what: `p_i = exp(z_i/τ) / Σ_j exp(z_j/τ) over ${fmtN(V)} entries.`, shapes: [['p', `${fmtN(V)} values per position`]],
      links: [['lm-head', 'input from'], [train ? 'loss' : 'sampling', 'output to']] },
  ];
  if (train) {
    const mem = trainingMemory(m, S), tc = trainingCompute(m, S), dpRows = byDim('dp');
    output.push(
      { id: 'loss', title: 'Cross-entropy loss', size: 'small', what: 'L = −(1/(B·T)) Σ log p(target token). Computed in FP32 on the last PP stage.', shapes: [['loss', 'scalar']], links: [['backward', 'starts']] },
      { id: 'backward', title: 'Backward pass', tag: 'reverse layer order',
        what: `For every linear y = x·W: dX = dY·Wᵀ and dW = Xᵀ·dY, about 2× the forward FLOPs. Uses saved activations${S.recompute === 'full' ? '; with full recompute, each layer\'s forward is re-run first (+1 forward)' : ''}. Weight gradients accumulate in an FP32 buffer across ${fmtN(Ly.micro)} micro-batches.`,
        shapes: [['FLOPs per step (all GPUs)', `${tc.c}·N·T = ${fmtFlops(tc.F)}`], ['per GPU, lower bound at peak', fmtTime(tc.tMin)]],
        hw: [['gradients', 'HBM', fmtBytes(mem.perStep[0].bytes), `FP32, ${mem.perStep[0].shardedBy}`], ['saved activations', 'HBM', fmtBytes(mem.perStep[1].bytes), 'freed layer by layer as backward proceeds']],
        tech: ['activation-checkpointing'], links: [['optimizer', 'then']] },
      ...(Ly.dpDense > 1 && S.zero < 3 ? [markerStage('coll-dp', `DP gradient sync (${S.zero === 0 ? 'all-reduce' : `ZeRO-${S.zero}: reduce-scatter + all-gather`})`,
        `${Ly.dpDense} data-parallel ranks hold the same weights and process different sequences; their gradients are summed across ranks. ${S.zero === 0 ? 'All-reduce leaves the full sum on every rank.' : 'Reduce-scatter leaves each rank 1/DP of the summed gradient; after the optimizer step, all-gather redistributes the updated BF16 weights.'}`,
        dpRows, ['data-parallel', 'zero', 'collectives'], [['optimizer', 'feeds']])] : []),
      ...(Ly.dpDense > 1 && S.zero === 3 ? [markerStage('coll-dp', 'DP gradient reduce-scatter (ZeRO-3)', 'Gradients are reduce-scattered so each rank keeps 1/DP of the summed gradient.', dpRows.filter((r) => r.op === 'reduce-scatter'), ['fsdp', 'collectives'], [['optimizer', 'feeds']])] : []),
      { id: 'optimizer', title: 'Optimizer step (AdamW)', size: 'small',
        what: 'Per parameter: m ← β₁m + (1−β₁)g, v ← β₂v + (1−β₂)g², w ← w − η(m̂/(√v̂ + ε) + λw), on the FP32 master copy, then cast to BF16. Element-wise and HBM-bandwidth-bound.',
        shapes: [['parameters updated on this GPU', fmtParams(Ly.denseShard / (S.zero >= 1 ? Ly.dpDense : 1) + Ly.expertShard / (S.zero >= 1 ? Ly.dpExpert : 1))]],
        hw: mem.persistent.slice(1).map((r) => [r.name, 'HBM', fmtBytes(r.bytes), `${r.dtype}, ${r.shardedBy}`]),
        tech: ['zero', 'mixed-precision'], links: [['embedding', 'next step starts at']] });
  } else {
    const st = inferenceSteps(m, S)[state.step];
    output.push({ id: 'sampling', title: 'Next-token selection', mono: true, viz: `"${esc(TOKENS.join(''))}<b>${esc(NEXT[0][0])}</b>"`,
      what: 'Greedy: argmax p. Sampling: draw from p (optionally top-k / top-p). The token is appended and the next decode step runs for one new position, reading the KV cache.',
      shapes: [[`${state.step}: FLOPs per replica`, fmtFlops(st.F)], ['HBM bytes per replica', fmtBytes(st.Q)], ['arithmetic intensity', `${st.I.toFixed(1)} FLOP/B (ridge ${inferenceSteps(m, S).ridge.toFixed(0)})`], ['bound', st.bound], ['time, lower bound', fmtTime(st.t)]],
      tech: ['prefill-decode', 'roofline', 'kv-cache'], links: [['softmax', 'input from'], ['kv-cache', 'appends to']] });
  }
  return { input, residual, layer, output, layerNote: f.kind === 'moe' && f.denseLayers ? `Layers 1–${f.denseLayers}: dense FFN (F = ${fmtN(f.denseInter)}); layers ${f.denseLayers + 1}–${L}: MoE.` : '' };
}

function templateStages(fam) {
  const g = { input: [], layer: [], output: [] };
  fam.outline.forEach(([grp, title], i) => g[grp].push({ id: `t${i}`, title, what: 'Template: explanation not written yet.', shapes: [], links: [] }));
  return { ...g, residual: null, layerNote: 'Template outline. Configs and numbers will be added per model.' };
}

// ---------------- data flow: render + selection ----------------
let byId = {}, order = [];

function boxHtml(s) {
  const cls = ['box', s.size || '', s.inner ? 'inner' : '', s.add ? 'add' : '', s.kind === 'coll' ? 'coll' : ''].filter(Boolean).join(' ');
  let body = s.viz ? `<div class="viz${s.mono ? ' mono pre' : ''}">${s.viz}</div>` : '';
  if (s.children && s.children.length) {
    const kids = s.children.map(boxHtml).join(s.row ? '' : '<div class="arrow">↓</div>');
    body += s.row ? `<div class="row"><div class="mini mono">${esc(s.pre)}</div>${kids}<div class="mini mono">${esc(s.post)}</div></div>` : `<div class="kids">${kids}</div>`;
  }
  return `<div class="${cls}" data-id="${s.id}"><div class="title">${esc(s.title)}${s.tag ? ` <span class="tag">${esc(s.tag)}</span>` : ''}</div>${body}</div>`;
}
const column = (list) => list.map(boxHtml).join('<div class="arrow">↓</div>');

function renderDataflow() {
  const tmpl = isTemplate();
  const st = tmpl ? templateStages(window.FAMILIES.find((f) => f.id === state.model)) : buildStages(currentModel(), state.S);
  byId = {}; order = [];
  const walk = (s) => { byId[s.id] = s; order.push(s.id); (s.children || []).forEach(walk); };
  st.input.forEach(walk); if (st.residual) walk(st.residual); st.layer.forEach(walk); st.output.forEach(walk);
  const L = tmpl ? 'L' : currentModel().layers;
  const sec3 = tmpl ? 'Output' : state.phase === 'training' ? 'Output, loss, backward, update' : 'Output';
  $('view').innerHTML = `<div id="diagram">
    <section class="group"><h2>1 · Input</h2>${column(st.input)}</section><div class="arrow">↓</div>
    <section class="group block"><h2>2 · ${tmpl ? 'Layer' : 'Transformer layer'} <span class="repeat">× ${L}</span></h2>${st.layerNote ? `<p class="group-note">${esc(st.layerNote)}</p>` : ''}
      <div class="block-body">${st.residual ? '<div class="residual" data-id="residual"><span class="title">residual stream h</span></div>' : ''}<div class="block-steps">${column(st.layer)}</div></div></section>
    <div class="arrow">↓</div>
    <section class="group"><h2>3 · ${sec3}</h2>${column(st.output)}
      ${!tmpl && state.phase === 'inference' ? '<div class="loop">↺ decode: the token is appended and one more position is computed, reading the KV cache</div>' : ''}
      ${!tmpl && state.phase === 'training' ? '<div class="loop">↺ next step: next global batch</div>' : ''}</section></div>`;
  if (!byId[state.stage]) state.stage = order[0];
  select(state.stage, false);
}

function select(id, scroll = true) {
  const s = byId[id]; if (!s) return;
  state.stage = id;
  const i = order.indexOf(id), nextId = i < order.length - 1 ? order[i + 1] : null;
  const related = new Set((s.links || []).map(([t]) => t));
  document.querySelectorAll('.badge').forEach((b) => b.remove());
  document.querySelectorAll('#view [data-id]').forEach((el) => {
    const e = el.dataset.id;
    el.classList.toggle('current', e === id); el.classList.toggle('next', e === nextId);
    el.classList.toggle('related', related.has(e) && e !== id && e !== nextId);
  });
  const mark = (e, t) => { const n = document.querySelector(`#view [data-id="${e}"] > .title`); if (n) n.insertAdjacentHTML('beforeend', `<span class="badge ${t.toLowerCase()}">${t}</span>`); };
  mark(id, 'CURRENT'); if (nextId) mark(nextId, 'NEXT');

  const table = (rows) => `<table>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join('')}</table>`;
  const hw = s.hw && s.hw.length ? `<h4>Residency (per GPU)</h4><table class="hw"><tr><th>tensor</th><th>where</th><th>bytes</th><th></th></tr>${s.hw.map((r) => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td class="mono">${esc(r[2])}</td><td class="note">${esc(r[3] || '')}</td></tr>`).join('')}</table>` : '';
  const coll = s.coll && s.coll.length ? `<h4>Collective (per GPU)</h4>${collList(s.coll)}` : (s.kind === 'coll' ? '<p class="note">No transfer at this degree.</p>' : '');
  const tech = s.tech && s.tech.length ? `<h4>Deep dives</h4><p class="techs">${s.tech.map(techLink).join('')}</p>` : '';
  const links = (s.links || []).filter(([t]) => byId[t]).map(([t, r]) => `<li>${esc(r)} <a data-go="${t}">${esc(byId[t].title)}</a></li>`).join('');
  $('panel').innerHTML = `<div class="panel-current">
      <div class="panel-kicker">CURRENT · ${i + 1} / ${order.length}</div><h3>${esc(s.title)}</h3><p>${esc(s.what)}</p>
      ${s.why ? `<h4>Detail</h4><p>${esc(s.why)}</p>` : ''}
      ${s.shapes && s.shapes.length ? `<h4>Shapes (${isTemplate() ? 'template' : esc(shapeCtx(currentModel(), state.S).label)})</h4>${table(s.shapes)}` : ''}
      ${hw}${coll}${tech}${links ? `<h4>Connections</h4><ul>${links}</ul>` : ''}</div>
    ${nextId ? `<button class="panel-next" data-go="${nextId}"><span class="panel-kicker">NEXT</span>${esc(byId[nextId].title)} ▶</button>` : ''}`;
  $('step-label').textContent = `${i + 1} / ${order.length}: ${s.title}`;
  if (scroll) document.querySelector(`#view [data-id="${id}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  writeHash();
}

// Narrow form for the side panel: one key/value block per collective.
function collList(rows) {
  const unit = S_unit();
  return rows.map((r) => `<table class="coll-kv">
    <tr><td>operation</td><td><b>${esc(r.op)}</b></td></tr><tr><td>where</td><td>${esc(r.when)}</td></tr>
    <tr><td>message</td><td class="mono">${fmtBytes(r.msg)}</td></tr><tr><td>count / ${unit}</td><td class="mono">${fmtN(r.count)}</td></tr>
    <tr><td>volume / GPU</td><td class="mono">${fmtBytes(r.volume)}</td></tr><tr><td>link</td><td class="${r.intra ? '' : 'inter'}">${esc(r.link)}</td></tr>
    <tr><td>time at link BW</td><td class="mono">${fmtTime(r.time)}</td></tr></table>`).join('') +
    '<p class="note">Ring-algorithm volume ÷ nominal per-direction link bandwidth, without overlap with compute.</p>';
}

function collTable(rows) {
  const unit = S_unit();
  return `<table class="coll-t"><tr><th>collective</th><th>where</th><th>message</th><th>count / ${unit}</th><th>volume / GPU</th><th>link</th><th>time at link BW</th></tr>
    ${rows.map((r) => `<tr><td>${esc(r.op)}</td><td>${esc(r.when)}</td><td class="mono">${fmtBytes(r.msg)}</td><td class="mono">${fmtN(r.count)}</td>
      <td class="mono">${fmtBytes(r.volume)}</td><td class="${r.intra ? '' : 'inter'}">${esc(r.link)}</td><td class="mono">${fmtTime(r.time)}</td></tr>`).join('')}</table>
    <p class="note">Ring-algorithm volume ÷ nominal per-direction link bandwidth, without overlap with compute.</p>`;
}

// ---------------- memory lens ----------------
function memRows(rows, cap) {
  return rows.map((r) => `<tr><td>${esc(r.name)}${r.approx ? ' <span class="approx">≈</span>' : ''}${r.tech ? ` ${techLink(r.tech)}` : ''}</td><td>${esc(r.dtype)}</td><td>${esc(r.shardedBy)}</td>
    <td class="mono num">${fmtBytes(r.bytes)}</td><td class="barcell"><span class="mbar" style="width:${Math.min(100, (100 * r.bytes) / cap)}%"></span></td>
    <td class="note">${esc(r.lifetime)}${r.detail ? `<br>${esc(r.detail)}` : ''}</td></tr>`).join('');
}

function kvChart(m, S) {
  const k = kvGrowth(m, S), xMax = Math.min(m.context, Math.max(2 * k.current, Math.ceil(k.maxCtx * 1.25), 1024));
  const W = 600, Hh = 200, padL = 56, padB = 28, padR = 40;
  const yMax = Math.max(k.cap, k.fixed + k.perTok * xMax) * 1.05;
  const sx = (x) => padL + (x / xMax) * (W - padL - padR), sy = (y) => Hh - padB - (y / yMax) * (Hh - padB - 10);
  const pts = [0, xMax].map((x) => `${sx(x)},${sy(k.fixed + k.perTok * x)}`).join(' ');
  const gib = (b) => (b / 2 ** 30).toFixed(0);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * xMax));
  let svg = `<svg width="${W}" height="${Hh}" role="img" aria-label="memory per GPU vs context length">`;
  [0, 0.5, 1].forEach((t) => { const y = t * yMax; svg += `<line x1="${padL}" x2="${W - padR}" y1="${sy(y)}" y2="${sy(y)}" stroke="#eef0f3"/><text x="${padL - 6}" y="${sy(y) + 4}" font-size="10" text-anchor="end" fill="#6b7280">${gib(y)} GiB</text>`; });
  ticks.forEach((x) => { svg += `<text x="${sx(x)}" y="${Hh - 10}" font-size="10" text-anchor="middle" fill="#6b7280">${fmtN(x)}</text>`; });
  svg += `<line x1="${padL}" x2="${W - padR}" y1="${sy(k.cap)}" y2="${sy(k.cap)}" stroke="#dc2626" stroke-dasharray="4 3"/><text x="${W - padR}" y="${sy(k.cap) - 5}" font-size="10" text-anchor="end" fill="#dc2626">HBM capacity ${(k.cap / 2 ** 30).toFixed(1)} GiB</text>`;
  svg += `<polyline points="${pts}" fill="none" stroke="#2563eb" stroke-width="2"/>`;
  const cx = Math.min(k.current, xMax);
  svg += `<circle cx="${sx(cx)}" cy="${sy(k.fixed + k.perTok * cx)}" r="5" fill="#2563eb" stroke="#fff" stroke-width="2"><title>${fmtN(k.current)} tokens: ${fmtBytes(k.fixed + k.perTok * k.current)}</title></circle>`;
  for (let x = 0; x <= xMax; x += xMax / 40) svg += `<rect x="${sx(x) - 6}" y="10" width="12" height="${Hh - padB - 10}" fill="transparent"><title>${fmtN(x)} tokens/sequence → ${fmtBytes(k.fixed + k.perTok * x)} per GPU</title></rect>`;
  svg += '</svg>';
  return `<div class="card"><h3>KV cache growth over sequence length</h3>
    <p>Per-GPU memory = weights + workspace (${fmtBytes(k.fixed)}) + KV cache (${fmtBytes(k.perTok)} per position for ${layout(m, S).bReplica} sequences). Current: ${fmtN(k.current)} tokens per sequence.
    ${k.maxCtx > 0 ? `Fits up to <b>${fmtN(k.maxCtx)}</b> tokens per sequence at this batch.` : '<b class="oom">Weights alone exceed HBM.</b>'}</p>${svg}
    <p class="note">x: tokens per sequence (prompt + generated). Hover for values.</p></div>`;
}

function stepTable(m, S) {
  const st = inferenceSteps(m, S), r = (k, f) => `<td class="mono">${f(st.prefill[k])}</td><td class="mono">${f(st.decode[k])}</td>`;
  return `<div class="card"><h3>Prefill vs decode (one replica, ${fmtN(S.tp)} TP GPU${S.tp > 1 ? 's' : ''})</h3>
    <table class="cmp"><tr><th></th><th>prefill (${fmtN(S.prompt)} positions)</th><th>decode (1 position, KV ${fmtN(S.prompt + S.output)})</th></tr>
    <tr><td>FLOPs</td>${r('F', fmtFlops)}</tr><tr><td>HBM bytes</td>${r('Q', fmtBytes)}</tr>
    <tr><td>arithmetic intensity</td>${r('I', (x) => `${x >= 100 ? fmtN(x) : x.toFixed(1)} FLOP/B`)}</tr>
    <tr><td>compute time (F / peak)</td>${r('tc', fmtTime)}</tr><tr><td>memory time (bytes / HBM BW)</td>${r('tm', fmtTime)}</tr>
    <tr><td><b>bound</b></td><td><b>${st.prefill.bound}</b></td><td><b>${st.decode.bound}</b></td></tr>
    <tr><td>tokens / s per replica</td>${r('tps', (x) => fmtN(x))}</tr></table>
    <p class="note">Ridge point ${gpuOf(S).name} (${st.dtype.toUpperCase()}): ${fmtFlops(st.pi)}/s ÷ ${fmtBW(st.beta)} = ${st.ridge.toFixed(0)} FLOP/B.
    Intensity above the ridge: compute-bound; below: memory-bandwidth-bound. Lower bounds at peak throughput, no overlap. ${techLink('roofline')} ${techLink('prefill-decode')}</p></div>`;
}

function renderMemory() {
  if (isTemplate()) { $('view').innerHTML = '<div class="card"><h3>Memory</h3><p>Template: no model configs yet.</p></div>'; return; }
  const m = currentModel(), S = state.S, train = S.phase === 'training';
  const mem = train ? trainingMemory(m, S) : inferenceMemory(m, S), G = gpuOf(S);
  const totalBar = `<div class="totalbar"><span style="width:${Math.min(100, (100 * mem.total) / mem.cap)}%" class="${mem.fits ? '' : 'oom'}"></span></div>`;
  const head = `<tr><th>tensor</th><th>dtype</th><th>sharded across</th><th>bytes / GPU</th><th>share of HBM</th><th>lifetime</th></tr>`;
  let html = `<div class="card"><h3>Per-GPU HBM residency: ${esc(G.name)}, ${fmtN(nGPUs(S))} GPUs</h3>
    <p>Total ${fmtBytes(mem.total)} of ${fmtBytes(mem.cap)} ${mem.fits ? '' : `<b class="oom">exceeds HBM by ${fmtBytes(mem.total - mem.cap)}</b>`}</p>${totalBar}
    <h4>Allocated at load (persistent): ${fmtBytes(mem.persistentTotal)}</h4><table class="mem">${head}${memRows(mem.persistent, mem.cap)}</table>
    <h4>Materialized ${train ? 'per step' : 'per request / per step'}: ${fmtBytes(mem.perStepTotal)}</h4><table class="mem">${head}${memRows(mem.perStep, mem.cap)}</table>
    <p class="note">≈: estimate (framework-dependent). ${train ? `Precision: BF16 weights, FP32 gradients, FP32 master weights, ${trainPrecision(m).momentDtype} AdamW moments.${m.trainPrecision ? ` ${esc(m.trainPrecision.note)}` : ''}` : 'Inference holds no gradients or optimizer states.'}</p></div>`;
  if (!train) html += kvChart(m, S) + stepTable(m, S);
  else {
    const tc = trainingCompute(m, S), Ly = layout(m, S);
    html += `<div class="card"><h3>Compute per step</h3><table class="cmp">
      <tr><td>tokens per step</td><td class="mono">${fmtN(tc.tokens)} (${fmtN(S.gbs)} sequences × ${fmtN(S.seq)})</td></tr>
      <tr><td>FLOPs per step (${tc.c}·N·T + attention)</td><td class="mono">${fmtFlops(tc.F)}</td></tr>
      <tr><td>per GPU</td><td class="mono">${fmtFlops(tc.perGPU)}</td></tr>
      <tr><td>time per step at peak BF16 (lower bound)</td><td class="mono">${fmtTime(tc.tMin)}</td></tr>
      <tr><td>micro-batches per DP rank</td><td class="mono">${fmtN(Ly.micro)}</td></tr>
      ${S.pp > 1 ? `<tr><td>pipeline bubble (1F1B)</td><td>${esc(bubbleText(S, Ly))}</td></tr>` : ''}</table></div>`;
  }
  html += `<div class="card"><h3>Transfers (per GPU)</h3><table class="cmp"><tr><th>when</th><th>path</th><th>what</th><th>bytes</th><th>link</th><th>time at link BW</th></tr>
    ${transfers(m, S).map((t) => `<tr><td>${esc(t.phase)}</td><td>${esc(t.path)}</td><td>${esc(t.what)}</td><td class="mono">${fmtBytes(t.bytes)}</td><td>${esc(t.link)}</td><td class="mono">${fmtTime(t.time)}</td></tr>`).join('')}</table>
    <p class="note">GPU-to-GPU transfers are listed in the Parallelism lens.</p></div>`;
  $('view').innerHTML = html;
}

// ---------------- parallelism lens ----------------
function renderParallel() {
  if (isTemplate()) { $('view').innerHTML = '<div class="card"><h3>Parallelism</h3><p>Template: no model configs yet.</p></div>'; return; }
  const m = currentModel(), S = state.S, Ly = layout(m, S), train = S.phase === 'training', coll = collectiveRows(m, S), moe = m.ffn.kind === 'moe';
  const lk = (d) => link(S, d);
  const card = (dim, title, degree, active, shards, replicates, note, techs) => {
    const rows = coll.filter((r) => r.dim === dim);
    return `<div class="card pcard ${active ? '' : 'inactive'}"><h3>${title} <span class="deg">${dim.toUpperCase()} = ${degree}</span> ${active ? `<span class="lk ${lk(dim).intra ? '' : 'inter'}">${esc(lk(dim).name)}</span>` : ''}</h3>
      <p><b>Shards:</b> ${shards}</p><p><b>Replicates:</b> ${replicates}</p><p>${note}</p>
      ${active ? (rows.length ? collTable(rows) : '<p class="note">No GPU-to-GPU transfer for this dimension in this phase.</p>') : `<p class="note">Inactive at degree 1: set ${dim.toUpperCase()} &gt; 1 to see its sharding and collectives.</p>`}
      <p class="techs">${techs.map(techLink).join('')}</p></div>`;
  };
  const zeroText = ['ZeRO-0: every DP rank holds full weights, gradients and optimizer states.', 'ZeRO-1: optimizer states (master weights, moments) split 1/DP.', 'ZeRO-2: optimizer states and gradients split 1/DP.', 'ZeRO-3 / FSDP: optimizer states, gradients and weights split 1/DP; weights are all-gathered per layer when used.'][S.zero];
  const inf = (t) => (train ? '' : `<b>Inference:</b> ${t}`);
  $('view').innerHTML = `<div class="card"><h3>Layout: ${fmtN(Ly.N)} GPUs = DP ${S.dp} × TP ${S.tp} × PP ${S.pp} × CP ${S.cp}${moe ? `, EP ${S.ep}` : ''} · ${fmtN(Ly.nodes)} node${Ly.nodes > 1 ? 's' : ''} of ${gpuOf(S).perNode} × ${esc(gpuOf(S).name)}</h3>
      <p>Rank order, innermost first: TP, CP, DP, PP. Inside a node the ${gpuOf(S).perNode} GPUs are connected all-to-all over NVLink: ${fmtBW(gpuOf(S).nvlinkPair)} per direction between any pair, ${fmtBW(gpuOf(S).nvlink)} per direction across all of one GPU's links. Collectives inside a node are computed at the all-links figure, P2P sends at the per-pair figure. A group that spans nodes is limited by the NIC (${esc(gpuOf(S).nicName)}, ${fmtBW(gpuOf(S).nic)} per GPU). Keep TP inside a node; CP and DP tolerate crossing nodes better.</p>
      ${gpuOf(S).assumed ? `<p class="note">Assumption: ${esc(gpuOf(S).assumed)}</p>` : ''}<p>
      ${train ? '' : 'In inference, TP and EP are the common choices inside a node; PP is used when weights exceed one node; DP runs independent replicas.'}</p></div>
    ${card('dp', train ? 'Data parallel (DP / ZeRO / FSDP)' : 'Data parallel (replicas)', S.dp, S.dp > 1,
      train ? `batch: each rank processes ${fmtN(S.gbs / S.dp)} of ${fmtN(S.gbs)} sequences per step. ${zeroText}` : `requests: each replica serves ${fmtN(Ly.bReplica)} of ${fmtN(S.batch)} sequences.`,
      train ? (S.zero === 3 ? 'nothing persistent; weights are re-gathered per layer' : `weights${S.zero < 2 ? ', gradients' : ''}${S.zero < 1 ? ', optimizer states' : ''}`) : 'all weights (each replica is a full copy)',
      train ? `${moe ? `Expert parameters are reduced over an expert-DP group of ${fmtN(Ly.dpExpert)} ranks. ` : ''}Gradient sync over ${fmtN(Ly.dpDense)} ranks (DP × CP) once per step.` : inf('no collective between replicas; throughput scales with DP.'),
      ['data-parallel', 'zero', 'fsdp'])}
    ${card('tp', 'Tensor parallel (TP, with SP)', S.tp, S.tp > 1,
      `attention heads (${attnDims(m).heads}/${S.tp} query heads per rank${attnDims(m).kv !== null ? `, ${Math.ceil(attnDims(m).kv / S.tp)} KV head(s)` : ''}), FFN intermediate dim (${fmtN(m.ffn.kind === 'dense' ? m.ffn.inter : m.ffn.expertInter)}/${S.tp}), vocabulary of embedding and LM head; with SP, norms and dropout along the sequence.`,
      m.attention.kind === 'MLA' ? 'MLA KV latent (c_KV, k_R) and its cache' : 'residual stream (without SP)',
      `Two collectives per layer in forward (and two in backward), on the critical path: usually kept inside one node. ${inf('common inside a node; also splits the KV cache by head.')}`,
      ['tensor-parallel', 'sequence-parallel'])}
    ${card('pp', 'Pipeline parallel (PP)', S.pp, S.pp > 1,
      `layers: ${S.pp} stages of ${Ly.layersPerStage} layers; each stage holds only its layers' weights${train ? ', gradients and optimizer states' : ' and KV cache'}.`, 'nothing',
      train ? `Micro-batches flow through stages (1F1B). ${bubbleText(S, Ly)} The first stage holds ${Ly.inflight} micro-batches of activations.` : inf('used when weights exceed one node; stages run sequentially per token, so latency is not reduced.'),
      ['pipeline-parallel'])}
    ${card('ep', 'Expert parallel (EP)', S.ep, moe && S.ep > 1,
      moe ? `routed experts: ${fmtN(Ly.expertsPerGPU)} of ${m.ffn.experts} per rank per layer.` : 'not applicable: dense model.',
      moe ? `attention and shared-expert weights (split by TP ${S.tp} only)` : '—',
      moe ? `Two all-to-all per MoE layer in forward (dispatch, combine) and two in backward. Expert load imbalance makes the slowest rank set the step time. The volume below assumes every routed token copy is sent separately in BF16 over the slowest link in the group; node-limited routing (DeepSeek-V3: at most 4 nodes per token, NVLink forwarding inside a node) and FP8 dispatch reduce inter-node bytes. ${inf('common for MoE inference.')}` : '',
      ['expert-parallel', 'moe'])}
    ${card('cp', 'Context parallel (CP)', S.cp, S.cp > 1 && (train || state.step === 'prefill'),
      `sequence: ${fmtN((train ? S.seq : S.prompt) / S.cp)} of ${fmtN(train ? S.seq : S.prompt)} positions per rank; activations ÷ CP.`,
      'weights (gradients are reduced over DP × CP)',
      `Ring attention: K, V blocks are passed between CP ranks so every query chunk is scored against all earlier keys. ${inf('used for long-prompt prefill; not used in decode (one new position).')}`,
      ['context-parallel'])}`;
}

// ---------------- main render + events ----------------
function render() {
  document.querySelectorAll('#models button').forEach((b) => b.classList.toggle('on', b.dataset.model === state.model));
  controls(); modelLine();
  const flow = state.lens === 'dataflow';
  document.body.classList.toggle('wide', !flow);
  $('stepper').hidden = !flow;
  if (flow) renderDataflow(); else { (state.lens === 'memory' ? renderMemory : renderParallel)(); writeHash(); }
}

document.addEventListener('click', (e) => {
  const mb = e.target.closest('#models button');
  if (mb) { state.model = mb.dataset.model; applyPreset(); return render(); }
  const set = e.target.closest('[data-set]');
  if (set) { state[set.dataset.set] = set.dataset.val; if (set.dataset.set === 'phase') applyPreset(); return render(); }
  if (e.target.closest('[data-preset]')) { applyPreset(); return render(); }
  const go = e.target.closest('[data-go]');
  if (go) return select(go.dataset.go);
  const box = e.target.closest('#view [data-id]');
  if (box) select(box.dataset.id, false);
});
document.addEventListener('change', (e) => {
  const k = e.target.dataset.s;
  if (!k) return;
  const v = e.target.value;
  state.S[k] = /^\d+$/.test(v) ? Math.max(1, Number(v)) : v;
  render();
});
$('prev').onclick = () => select(order[(order.indexOf(state.stage) - 1 + order.length) % order.length]);
$('next').onclick = () => select(order[(order.indexOf(state.stage) + 1) % order.length]);
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select') || state.lens !== 'dataflow') return;
  if (e.key === 'ArrowRight') $('next').click();
  if (e.key === 'ArrowLeft') $('prev').click();
});

modelBar();
readHash();
render();
