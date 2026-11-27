// Helpers shared by the chapters: escaping, SVG figures, scales, and a seeded random generator
// for illustrative examples (for instance which experts the example tokens select).

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// GPUs per node for the serving chapters (KV cache, inference engines); shared so both place TP alike.
window.SERVE = { gpn: 8 };
const gpnSelect = (id) => `<label class="nw">GPUs per node <select id="${id}">${[4, 8].map((v) => `<option${v === SERVE.gpn ? ' selected' : ''}>${v}</option>`).join('')}</select></label>`;
const bindGpn = (root, id, rerender) => root.querySelector(`#${id}`).addEventListener('change', (e) => { SERVE.gpn = Number(e.target.value); rerender(); });

// Link to a model's config.json. A gated repository needs a Hugging Face login and acceptance of the license;
// with `detail`, the model's sourceNote says where the values come from.
const configLink = (m, label = 'config.json', detail = false) => {
  const notes = [m.gated ? 'gated: requires a Hugging Face login and acceptance of the license' : '', detail ? m.sourceNote ?? '' : ''].filter(Boolean);
  return `<a href="${esc(m.source)}" target="_blank" rel="noreferrer">${esc(m.name)} ${esc(label)}</a>${notes.length ? ` (${esc(notes.join('. '))})` : ''}`;
};
const showTok = (t) => esc(t.replace(/^ /, '␣'));   // a leading space is part of the token
const range = (n) => Array.from({ length: n }, (_, i) => i);

// SVG with a fixed coordinate system that scales down to the column width.
const figure = (body, w, h, label) =>
  `<svg class="fig" viewBox="0 0 ${w} ${h}" style="max-width:${w}px" role="img" aria-label="${esc(label)}">${body}</svg>`;

const logScale = (d0, d1, r0, r1) => (x) => r0 + ((Math.log10(x) - Math.log10(d0)) / (Math.log10(d1) - Math.log10(d0))) * (r1 - r0);

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hashStr(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
function shuffled(arr, rnd) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

// Native <select> for a dtype choice; `opts` are keys of DTYPE_BYTES.
const dtypeSelect = (name, value, opts) =>
  `<select data-dtype="${name}">${opts.map((o) => `<option value="${o}"${o === value ? ' selected' : ''}>${DTYPE_NAME[o]}</option>`).join('')}</select>`;

// Optional per-GPU speed figures, shared by the chapters that can turn FLOPs and bytes into times.
// Blank by default: the page stays hardware-independent unless the reader enters them.
window.HW_OPT = window.HW_OPT || { tflops: '', tbps: '' };
const hwInputs = () => `<p class="controls">Optional, for times: <label class="nw">peak <input data-hw="tflops" type="number" min="0" step="any" value="${esc(window.HW_OPT.tflops)}" class="num-in" placeholder="—"> TFLOP/s</label> and <label class="nw">memory bandwidth <input data-hw="tbps" type="number" min="0" step="any" value="${esc(window.HW_OPT.tbps)}" class="num-in" placeholder="—"> TB/s</label> per GPU, from your GPU's data sheet</p>`;
function bindHw(root, rerender) {
  root.querySelectorAll('[data-hw]').forEach((el) => el.addEventListener('change', () => { window.HW_OPT[el.dataset.hw] = el.value; rerender(); }));
}
const hwPeak = () => Number(window.HW_OPT.tflops) * 1e12, hwBw = () => Number(window.HW_OPT.tbps) * 1e12;
const fmtTime = (s) => (s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(s >= 0.1 ? 0 : 2)} ms` : `${(s * 1e6).toFixed(0)} µs`);
