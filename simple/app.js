// How an LLM turns text into the next word: one page, whole picture.
// Model sizes are Llama 3.1 8B's published config (huggingface.co/meta-llama/Llama-3.1-8B, config.json).
// The example token IDs, attention weights and probabilities are ILLUSTRATIVE, not real model output.

const MODEL = {
  name: 'Llama 3.1 8B',
  layers: 32, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128,
  ffn: 14336, vocab: 128256, context: 131072,
};

const TOKENS = ['The', ' cat', ' sat', ' on', ' the'];
const IDS = [791, 8415, 7731, 389, 279]; // illustrative
const NEXT = [ // illustrative next-token probabilities, top 5
  [' mat', 0.41], [' floor', 0.17], [' couch', 0.08], [' bed', 0.06], [' sofa', 0.05],
];

// ---------- numbers derived from the config ----------
const M = MODEL, T = TOKENS.length;
const fmt = (n) => n.toLocaleString('en-US');
const attnParams = 2 * M.hidden * M.heads * M.headDim + 2 * M.hidden * M.kvHeads * M.headDim;
const mlpParams = 3 * M.hidden * M.ffn;
const layerParams = attnParams + mlpParams + 2 * M.hidden;
const embedParams = M.vocab * M.hidden;
const totalParams = 2 * embedParams + M.layers * layerParams + M.hidden; // embedding + LM head are separate
const pct = (x) => (100 * x / totalParams).toFixed(0) + '%';
const millions = (n) => (n / 1e6).toFixed(0) + ' million';
const billions = (n) => (n / 1e9).toFixed(2) + ' billion';

// ---------- what each box means ----------
const STAGES = {
  text: {
    title: 'Text prompt',
    what: 'Your input. The model never sees letters: everything after this works on numbers.',
    shapes: [['in', `"${TOKENS.join('')}"`]],
    links: [['tokenizer', 'is split into tokens by']],
  },
  tokenizer: {
    title: 'Tokenizer',
    what: 'Splits text into tokens, the pieces the model knows. Common words are a single token, while rare words split into several. The space usually belongs to the following word ("␣cat").',
    why: `Llama 3 uses byte-level BPE with a fixed vocabulary of ${fmt(M.vocab)} tokens. Any text can be encoded, because unknown words fall back to smaller pieces.`,
    shapes: [['in', 'text'], ['out', `${T} tokens`]],
    links: [['token-ids', 'each token becomes a number'], ['lm-head', 'uses the same vocabulary at the output end']],
  },
  'token-ids': {
    title: 'Token IDs',
    what: 'Each token is replaced by its position in the vocabulary, an integer between 0 and ' + fmt(M.vocab - 1) + '.',
    shapes: [['out', `${T} integers`]],
    note: 'The IDs shown are illustrative.',
    links: [['embedding', 'each ID selects one row of']],
  },
  embedding: {
    title: 'Embedding lookup',
    what: `A learned table with one row per vocabulary token: ${fmt(M.vocab)} rows × ${fmt(M.hidden)} numbers. Token ID n simply picks row n, so there is no computation, only a lookup.`,
    why: 'A number like 8415 carries no meaning. A vector of 4,096 learned numbers can place tokens with similar meanings close together.',
    shapes: [['in', `${T} IDs`], ['out', `${T} × ${fmt(M.hidden)} numbers`], ['table size', `${millions(embedParams)} parameters (${pct(embedParams)} of the model)`]],
    links: [['residual', 'these vectors start']],
  },
  residual: {
    title: 'Residual stream',
    what: `The running state: ${T} vectors of ${fmt(M.hidden)} numbers, one per token. It enters the first layer, passes through all ${M.layers} layers, and leaves at the end. Each layer reads it, computes something, and adds the result back. Nothing is replaced, only added.`,
    why: 'Because layers only add, information from the embedding can travel straight to the output, and every layer can refine what earlier layers wrote. This also keeps training a 32-layer stack stable.',
    shapes: [['size', `${T} × ${fmt(M.hidden)} numbers, constant through every layer`]],
    links: [['embedding', 'starts as the output of'], ['add1', 'attention writes into it at'], ['add2', 'the MLP writes into it at'], ['final-norm', 'is finally read by']],
  },
  norm1: {
    title: 'RMSNorm (before attention)',
    what: 'Rescales each token\'s vector to a steady overall size, then multiplies by learned per-dimension weights.',
    why: 'After many additions the stream\'s values can grow large. Normalising the input of each sub-layer keeps the arithmetic well-behaved. The stream itself is not changed: only the copy fed to attention is.',
    shapes: [['in / out', `${T} × ${fmt(M.hidden)}`]],
    links: [['residual', 'reads from'], ['attention', 'feeds']],
  },
  attention: {
    title: 'Self-attention',
    what: `The only place where tokens exchange information. Each token produces a query ("what am I looking for"), a key ("what do I contain") and a value ("what I pass on"). A score query·key is computed for every pair of tokens. A causal mask hides later tokens, softmax turns each row of scores into weights that sum to 1, and each token's output is the weighted sum of the values.`,
    why: `This runs as ${M.heads} heads in parallel, each ${M.headDim} numbers wide, so different heads can track different relations. Llama shares ${M.kvHeads} key/value heads among the ${M.heads} query heads (grouped-query attention) to save memory. Word order enters here: RoPE rotates queries and keys by their position, so scores depend on how far apart two tokens are.`,
    shapes: [
      ['in', `${T} × ${fmt(M.hidden)}`],
      ['queries', `${T} × ${M.heads} heads × ${M.headDim}`],
      ['keys, values', `${T} × ${M.kvHeads} heads × ${M.headDim} each`],
      ['scores', `${M.heads} heads × ${T} × ${T}  (grows with length²)`],
      ['out', `${T} × ${fmt(M.hidden)}`],
      ['parameters', `${millions(attnParams)} per layer`],
    ],
    note: 'The grid shows one head\'s weights for the example, with darker meaning more attention. Values are illustrative.',
    links: [['norm1', 'reads the normalised stream from'], ['add1', 'its output is added back at'], ['softmax', 'uses the same softmax as the output step'], ['sampling', 'keeps its keys and values cached while generating (KV cache)']],
  },
  add1: {
    title: 'Add (residual connection)',
    what: 'stream = stream + attention(norm(stream)). Attention\'s result is added to the residual stream rather than replacing it.',
    shapes: [['out', `${T} × ${fmt(M.hidden)}`]],
    links: [['residual', 'updates'], ['norm2', 'then feeds']],
  },
  norm2: {
    title: 'RMSNorm (before MLP)',
    what: 'The same normalisation as before attention, with its own learned weights, applied to the copy fed to the MLP.',
    shapes: [['in / out', `${T} × ${fmt(M.hidden)}`]],
    links: [['mlp', 'feeds']],
  },
  mlp: {
    title: 'Feed-forward network (MLP)',
    what: `Processes each token separately, with no mixing between tokens. Two projections widen the vector from ${fmt(M.hidden)} to ${fmt(M.ffn)} numbers (called "gate" and "up"). The activation function is applied to the gate, the two are multiplied element-wise (SwiGLU), and a third projection shrinks the result back to ${fmt(M.hidden)}.`,
    why: `Attention moves information between tokens; the MLP transforms it. Much of what the model "knows" is thought to be stored here, and it holds most of the parameters: ${millions(mlpParams)} per layer, ${pct(M.layers * mlpParams)} of the whole model.`,
    shapes: [['in', `${T} × ${fmt(M.hidden)}`], ['widened', `${T} × ${fmt(M.ffn)}  (×2: gate and up)`], ['out', `${T} × ${fmt(M.hidden)}`]],
    links: [['activation', 'contains'], ['add2', 'its output is added back at'], ['attention', 'complements: attention mixes tokens, the MLP does not']],
  },
  activation: {
    title: 'Activation function (SiLU)',
    what: 'SiLU(x) = x · sigmoid(x). It is applied to each of the 14,336 numbers individually. Positive values pass through almost unchanged, while negative values are squashed smoothly toward zero.',
    why: 'This is the non-linear step. Without it, the two projections around it would multiply into a single linear map, and stacking 32 layers would still be one linear map. Non-linearity is what lets depth add expressive power.',
    shapes: [['in / out', `${T} × ${fmt(M.ffn)}`]],
    links: [['mlp', 'sits in the middle of']],
  },
  add2: {
    title: 'Add (residual connection)',
    what: `stream = stream + mlp(norm(stream)). This completes one layer. The updated stream goes into the next layer, and the whole box repeats ${M.layers} times, each time with different weights.`,
    shapes: [['out', `${T} × ${fmt(M.hidden)}`]],
    links: [['residual', 'updates'], ['norm1', 'next layer starts again at'], ['final-norm', 'after the last layer, goes to']],
  },
  'final-norm': {
    title: 'Final RMSNorm',
    what: 'One last normalisation of the residual stream after the final layer.',
    shapes: [['in / out', `${T} × ${fmt(M.hidden)}`]],
    links: [['residual', 'reads the final state of'], ['lm-head', 'feeds']],
  },
  'lm-head': {
    title: 'LM head → logits',
    what: `Multiplies the last token's vector (${fmt(M.hidden)} numbers) by a ${fmt(M.hidden)} × ${fmt(M.vocab)} matrix, giving one score (a logit) for every token in the vocabulary. Only the last position is needed, because it is the one predicting what comes next.`,
    why: 'A higher logit means the model rates that token as a more likely continuation. Logits are unbounded and do not sum to anything yet.',
    shapes: [['in', `1 × ${fmt(M.hidden)}  (last token)`], ['out', `${fmt(M.vocab)} logits`], ['parameters', `${millions(embedParams)}`]],
    note: 'Logits shown are illustrative.',
    links: [['tokenizer', 'one score per entry of the vocabulary defined by'], ['softmax', 'scores go to']],
  },
  softmax: {
    title: 'Softmax → probabilities',
    what: `p(token) = e^logit / Σ e^logit over all ${fmt(M.vocab)} tokens. Every probability is positive and they sum to 1, so the output is a probability distribution over the vocabulary.`,
    why: 'Temperature divides the logits before softmax. A lower temperature makes the distribution sharper (more predictable text) and a higher one makes it flatter (more varied).',
    shapes: [['out', `${fmt(M.vocab)} probabilities summing to 1`]],
    note: 'The probabilities shown are illustrative. The rest of the vocabulary shares the remaining probability.',
    links: [['lm-head', 'turns the logits from'], ['attention', 'is the same operation attention uses for its weights'], ['sampling', 'feeds']],
  },
  sampling: {
    title: 'Pick the next token',
    what: 'Choose one token: the most likely (greedy), or a random draw weighted by probability (sampling, often limited to the top few). The chosen token is appended to the input and the whole process runs again. Repeating this loop is how text gets generated, one token at a time.',
    why: `Earlier tokens' keys and values do not change, so they are cached (the KV cache). Each new step then only computes the new token, at the cost of memory that grows with the text: ${fmt(2 * M.layers * M.kvHeads * M.headDim * 2 / 1024)} KiB per token in BF16 for this model.`,
    shapes: [['out', '1 token, then repeat']],
    links: [['softmax', 'chooses from'], ['tokenizer', 'the new token joins the input, as if tokenized'], ['attention', 'the KV cache stores keys and values from']],
  },
};

const ORDER = ['text', 'tokenizer', 'token-ids', 'embedding', 'residual', 'norm1', 'attention', 'add1',
  'norm2', 'mlp', 'activation', 'add2', 'final-norm', 'lm-head', 'softmax', 'sampling'];

// ---------- small visuals inside the boxes ----------
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const show = (t) => esc(t.replace(/^ /, '␣'));

$('model-line').textContent =
  `Sizes are ${M.name}'s real config: ${M.layers} layers, ${fmt(M.hidden)}-number vectors, ${fmt(M.vocab)}-token vocabulary, ${billions(totalParams)} parameters. ` +
  'Example values (token IDs, attention weights, probabilities) are illustrative.';
$('repeat').textContent = `× ${M.layers} layers`;
$('viz-text').textContent = `"${TOKENS.join('')}"`;
$('viz-tokens').innerHTML = TOKENS.map((t) => `<span class="chip">${show(t)}</span>`).join('');
$('viz-ids').textContent = `[${IDS.join(', ')}]`;
$('viz-embed').innerHTML = TOKENS.map((t, i) =>
  `${show(t).padEnd(5)} → [${[0.12, -0.48, 0.91, 0.05].map((v) => (v * (i % 2 ? -1 : 1) + i * 0.03).toFixed(2)).join(', ')}, … ${fmt(M.hidden)} numbers]`
).join('<br>');
$('viz-up').textContent = `${fmt(M.hidden)} → ${fmt(M.ffn)}`;
$('viz-down').textContent = `→ ${fmt(M.hidden)}`;
$('viz-logits').textContent = `[ ${NEXT.slice(0, 3).map(([t, p]) => `${(17.2 + Math.log(p / NEXT[0][1])).toFixed(1)} "${t.trim()}"`).join(', ')}, … ${fmt(M.vocab)} scores ]`;
$('viz-next').innerHTML = `"${esc(TOKENS.join(''))}<b>${esc(NEXT[0][0])}</b>"`;

// Attention weights for one head: illustrative raw scores, causal mask, softmax per row.
function attentionGrid() {
  const raw = [
    [0, 0, 0, 0, 0],
    [1.0, 1.5, 0, 0, 0],
    [0.3, 2.2, 1.0, 0, 0],
    [0.2, 1.0, 1.8, 1.2, 0],
    [0.1, 2.0, 0.8, 1.6, 0.9],
  ];
  const w = raw.map((row, i) => {
    const e = row.map((s, j) => (j <= i ? Math.exp(s) : 0));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  });
  const cell = 36;
  const left = 44, top = 30;
  let svg = `<svg width="${left + cell * T + 4}" height="${top + cell * T + 4}" role="img" aria-label="attention weights">`;
  svg += `<text x="${left}" y="10" font-size="10" fill="#8a94a3">looks at →</text>`;
  TOKENS.forEach((t, j) => {
    svg += `<text x="${left + j * cell + cell / 2}" y="${top - 6}" font-size="11" text-anchor="middle" fill="#5b6573">${show(t)}</text>`;
  });
  TOKENS.forEach((t, i) => {
    svg += `<text x="${left - 6}" y="${top + i * cell + cell / 2 + 4}" font-size="11" text-anchor="end" fill="#5b6573">${show(t)}</text>`;
    w[i].forEach((v, j) => {
      const masked = j > i;
      const fill = masked ? '#f1f3f5' : `rgba(37, 99, 235, ${0.08 + 0.92 * v})`;
      const tip = masked ? `"${t.trim()}" cannot see the later token "${TOKENS[j].trim()}" (causal mask)`
        : `"${t.trim()}" → "${TOKENS[j].trim()}": weight ${v.toFixed(2)}`;
      svg += `<rect x="${left + j * cell + 1}" y="${top + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="${fill}"><title>${esc(tip)}</title></rect>`;
    });
  });
  svg += '</svg>';
  return `<div style="display:flex;gap:14px;align-items:center">${svg}
    <div class="note">Each row is one token looking back at earlier tokens.<br>Rows sum to 1. Grey cells are hidden by the causal mask.<br>Hover a cell for its weight.</div></div>`;
}
$('viz-attn').innerHTML = attentionGrid();

// SiLU curve with the identity line for reference.
function siluChart() {
  const W = 220, H = 110, x0 = -5, x1 = 4, y0 = -1, y1 = 4;
  const sx = (x) => ((x - x0) / (x1 - x0)) * W;
  const sy = (y) => H - ((y - y0) / (y1 - y0)) * H;
  const pts = [];
  for (let x = x0; x <= x1 + 1e-9; x += 0.1) pts.push(`${sx(x).toFixed(1)},${sy(x / (1 + Math.exp(-x))).toFixed(1)}`);
  return `<div style="display:flex;gap:14px;align-items:center">
    <svg width="${W}" height="${H}" role="img" aria-label="SiLU activation curve">
      <line x1="0" y1="${sy(0)}" x2="${W}" y2="${sy(0)}" stroke="#d5dae1"/>
      <line x1="${sx(0)}" y1="0" x2="${sx(0)}" y2="${H}" stroke="#d5dae1"/>
      <line x1="${sx(-1)}" y1="${sy(-1)}" x2="${sx(4)}" y2="${sy(4)}" stroke="#b9c0ca" stroke-dasharray="3 3"/>
      <polyline points="${pts.join(' ')}" fill="none" stroke="#2563eb" stroke-width="2"/>
      <text x="${W - 4}" y="${sy(0) - 4}" font-size="10" text-anchor="end" fill="#8a94a3">x</text>
    </svg>
    <div class="note">Blue: SiLU(x) = x·sigmoid(x).<br>Dashed: y = x, i.e. no activation.<br>Negatives bend toward 0. That bend is the non-linearity.</div></div>`;
}
$('viz-silu').innerHTML = siluChart();

// Probabilities of the top candidates, one series.
function probBars() {
  const rest = 1 - NEXT.reduce((a, [, p]) => a + p, 0);
  const rows = [...NEXT, ['(all other tokens)', rest]];
  return rows.map(([t, p], i) => `
    <div style="display:flex;align-items:center;gap:8px;margin:3px 0" title="${esc(t)}: ${(p * 100).toFixed(0)}%">
      <span class="mono" style="width:130px;text-align:right">${i < NEXT.length ? `"${esc(t.trim())}"` : esc(t)}</span>
      <span style="flex:1;max-width:220px;background:#f1f3f5;border-radius:4px">
        <span style="display:block;height:12px;width:${p * 100}%;background:${i < NEXT.length ? '#2563eb' : '#b9c0ca'};border-radius:0 4px 4px 0"></span>
      </span>
      <span class="mono" style="width:40px">${(p * 100).toFixed(0)}%</span>
    </div>`).join('');
}
$('viz-probs').innerHTML = probBars();

// ---------- selection, related boxes, explanation panel ----------
const boxes = () => document.querySelectorAll('[data-id]');
let current = 0;

function select(id, scroll = true) {
  const s = STAGES[id];
  if (!s) return;
  current = ORDER.indexOf(id);
  const related = new Set(s.links.map(([to]) => to));
  boxes().forEach((el) => {
    el.classList.toggle('selected', el.dataset.id === id);
    el.classList.toggle('related', related.has(el.dataset.id));
  });
  const rows = s.shapes.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join('');
  const links = s.links.map(([to, phrase]) =>
    `<li>${esc(phrase)} <a data-go="${to}">${esc(STAGES[to].title)}</a></li>`).join('');
  $('panel').innerHTML = `
    <h3>${esc(s.title)}</h3>
    <p>${esc(s.what)}</p>
    ${s.why ? `<h4>Why it matters</h4><p>${esc(s.why)}</p>` : ''}
    <h4>Shapes in ${esc(M.name)} (${T}-token example)</h4>
    <table>${rows}</table>
    ${s.note ? `<p class="note" style="margin-top:6px">${esc(s.note)}</p>` : ''}
    <h4>How it connects</h4>
    <ul>${links}</ul>`;
  $('step-label').textContent = `Step ${current + 1} of ${ORDER.length}: ${s.title}`;
  if (scroll) document.querySelector(`[data-id="${id}"]`).scrollIntoView({ block: 'center', behavior: 'smooth' });
}

document.addEventListener('click', (e) => {
  const go = e.target.closest('[data-go]');
  if (go) return select(go.dataset.go);
  const box = e.target.closest('[data-id]');
  if (box) { e.stopPropagation(); select(box.dataset.id, false); }
});
$('prev').onclick = () => select(ORDER[(current - 1 + ORDER.length) % ORDER.length]);
$('next').onclick = () => select(ORDER[(current + 1) % ORDER.length]);
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') $('next').click();
  if (e.key === 'ArrowLeft') $('prev').click();
});

select('text', false);
