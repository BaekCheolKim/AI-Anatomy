// Page frame: model and GPU inputs in the top bar, the chapter list, URL-hash state, chapter rendering.
// Each chapter file in chapters/ registers { render(ctx) → HTML, mount(root, ctx, rerender) } in
// window.CHAPTERS. Chapters listed here without a file are shown as planned (see PLAN.md).

const OUTLINE = [
  ['Overview', [['overview', 'The whole process']]],
  ['Pre-training', [
    ['data', 'Data'], ['tokenization', 'Tokenization'], ['io', 'Input and output'], ['network', 'Inside the network'],
    ['attention', 'Attention variants'], ['moe', 'Mixture of experts'], ['images', 'Images as tokens'],
    ['training', 'Training loop'], ['scaling', 'Parameters and training tokens'], ['long-context', 'Long context'],
    ['parallel', 'Splitting across GPUs'], ['sampling', 'Sampling from the base model']]],
  ['Post-training', [
    ['chat', 'Conversations as tokens'], ['lora', 'Fine-tuning: full or LoRA'], ['distillation', 'Distillation'],
    ['context', 'Context, tools and thinking']]],
  ['Reinforcement learning', [
    ['rl', 'RL with checkable answers'], ['rl-gpus', 'One RL step on GPUs'], ['preferences', 'Preference training']]],
  ['Running a model', [
    ['prefill-decode', 'Prefill and decode'], ['kv-cache', 'KV cache and recurrent state'], ['precision', 'Precision and quantization'],
    ['fit', 'Will it fit?'], ['engines', 'Inference engines'], ['serving', 'Serving many requests'],
    ['request', 'From request to response'], ['agents', 'Agents']]],
  ['Other model families', [
    ['embeddings', 'Embedding models and retrieval'], ['diffusion', 'Image generation'], ['speech', 'Speech recognition']]],
];

const CHAPTERS_FLAT = OUTLINE.flatMap(([part, list]) => list.map(([id, title]) => ({ id, title, part })));
const DEFAULTS = { ch: 'overview', model: 'deepseek-v3', mem: 80, gpus: 8 };
const state = { ...DEFAULTS };
const $ = (s) => document.querySelector(s);
const built = (id) => Boolean(window.CHAPTERS && window.CHAPTERS[id]);
const modelById = (id) => window.MODELS.find((m) => m.id === id);

// ---------------- URL hash ----------------
// Unknown chapter or model ids (for example from links made before a model was replaced) fall back
// to the defaults instead of failing.
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const mem = Number(q.get('mem')), gpus = Number(q.get('gpus'));
  state.ch = CHAPTERS_FLAT.some((c) => c.id === q.get('ch')) ? q.get('ch') : DEFAULTS.ch;
  state.model = modelById(q.get('model')) ? q.get('model') : DEFAULTS.model;
  state.mem = mem > 0 ? mem : DEFAULTS.mem;
  state.gpus = Number.isInteger(gpus) && gpus > 0 ? gpus : DEFAULTS.gpus;
}
const hashOf = (over = {}) => `#${new URLSearchParams({ ...state, ...over })}`;

// ---------------- rendering ----------------
function renderBar() {
  $('#models').innerHTML = window.MODELS.map((m) =>
    `<a href="${hashOf({ model: m.id })}" data-model="${m.id}" class="${m.id === state.model ? 'on' : ''}" title="${esc(m.name)}">${esc(m.short)}</a>`).join('');
  if (document.activeElement !== $('#mem')) $('#mem').value = state.mem;
  if (document.activeElement !== $('#gpus')) $('#gpus').value = state.gpus;
}

function renderToc() {
  $('#toc').innerHTML = OUTLINE.map(([part, list]) => `<h4>${esc(part)}</h4><ol>${list.map(([id, title]) =>
    `<li><a href="${hashOf({ ch: id })}" data-ch="${id}" class="${id === state.ch ? 'on' : ''}${built(id) ? '' : ' planned'}">${esc(title)}</a></li>`).join('')}</ol>`).join('')
    + (CHAPTERS_FLAT.some((c) => !built(c.id)) ? '<p class="legend">Gray: planned, not built yet.</p>' : '');
}

function pager(i) {
  const prev = CHAPTERS_FLAT[i - 1], next = CHAPTERS_FLAT[i + 1];
  const link = (c, dir) => (c ? `<a href="${hashOf({ ch: c.id })}" data-ch="${c.id}" class="${built(c.id) ? '' : 'planned'}">${dir === 'prev' ? '← ' : ''}${esc(c.title)}${dir === 'next' ? ' →' : ''}</a>` : '<span></span>');
  return `<nav class="pager" aria-label="Previous and next chapter">${link(prev, 'prev')}${link(next, 'next')}</nav>`;
}

// Figures are drawn at the column's width so that their text keeps its size on narrow screens.
const columnWidth = () => Math.round(Math.min(720, Math.max(280, $('#chapter').clientWidth || 720)));
let drawnWidth = 0;

function renderChapter() {
  const i = CHAPTERS_FLAT.findIndex((c) => c.id === state.ch), c = CHAPTERS_FLAT[i], root = $('#chapter');
  drawnWidth = columnWidth();
  const ctx = { m: modelById(state.model), mem: state.mem * 1e9, gpus: state.gpus, width: drawnWidth };
  const head = `<p class="part">${esc(c.part)}</p><h1>${esc(c.title)}</h1>`;
  const body = built(c.id) ? window.CHAPTERS[c.id].render(ctx) : '<p class="lead planned">Planned. This chapter is not built yet; PLAN.md lists what it will show.</p>';
  root.innerHTML = head + body + pager(i);
  if (built(c.id) && window.CHAPTERS[c.id].mount) window.CHAPTERS[c.id].mount(root, ctx, rerenderChapter);
}

function rerenderChapter() { renderChapter(); }

function render() {
  renderBar(); renderToc(); renderChapter();
  history.replaceState(null, '', hashOf());
  document.title = `${CHAPTERS_FLAT.find((c) => c.id === state.ch).title} · AI-Anatomy`;
}

function go(over, scrollTop) {
  Object.assign(state, over);
  history.pushState(null, '', hashOf());
  render();
  if (scrollTop) window.scrollTo(0, 0);
}

// ---------------- events ----------------
document.addEventListener('click', (e) => {
  const mdl = e.target.closest('[data-model]');
  if (mdl) { e.preventDefault(); return go({ model: mdl.dataset.model }); }
  const ch = e.target.closest('[data-ch]');
  if (ch) {
    e.preventDefault();
    if (window.matchMedia('(max-width: 900px)').matches) $('#tocwrap').open = false;
    return go({ ch: ch.dataset.ch }, true);
  }
  if (e.target.closest('[data-home]')) { e.preventDefault(); return go({ ch: DEFAULTS.ch }, true); }
});

['mem', 'gpus'].forEach((k) => $(`#${k}`).addEventListener('input', (e) => {
  const v = Number(e.target.value);
  if (!(v > 0) || (k === 'gpus' && !Number.isInteger(v))) return;
  state[k] = v;
  history.replaceState(null, '', hashOf());
  renderChapter();
}));

document.addEventListener('keydown', (e) => {
  if ((e.target instanceof Element && e.target.matches('input, select, textarea')) || e.metaKey || e.ctrlKey || e.altKey) return;
  const i = CHAPTERS_FLAT.findIndex((c) => c.id === state.ch);
  if (e.key === 'ArrowLeft' && i > 0) go({ ch: CHAPTERS_FLAT[i - 1].id }, true);
  if (e.key === 'ArrowRight' && i < CHAPTERS_FLAT.length - 1) go({ ch: CHAPTERS_FLAT[i + 1].id }, true);
});

window.addEventListener('popstate', () => { readHash(); render(); });

let resizeTimer = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (Math.abs(columnWidth() - drawnWidth) > 4) renderChapter(); }, 120);
});

if (window.matchMedia('(max-width: 900px)').matches) $('#tocwrap').open = false;
readHash();
render();
