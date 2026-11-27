// Chapter: KV cache and recurrent state.
// Figures: (1) one GPU's memory: weights, workspace, and the KV pool that remains, with the tokens
// and sequences it holds; (2) measured next to computed.
(function () {
  let wprec = 'release', kvDtype = 'bf16', util = 0.9, seqLen = 32768;

  function figPool(m, ctx, W) {
    const G = ctx.gpus, mem = ctx.mem, { tp, pp, used, why } = servingLayout(m, G, SERVE.gpn);
    const Wt = weightBytes(m, wprec === 'release' ? {} : { experts: wprec, linear: wprec }).total, wPer = Wt / used, work = 0.05 * mem;
    const pool = Math.max(0, util * mem - wPer - work);
    const perTok = kvBytesPerTokenPerGPU(m, tp, kvDtype) / pp, fixed = cacheBytesPerGPU(m, 0, tp, kvDtype) / pp;
    const tokens = perTok > 0 ? Math.floor(pool / perTok) : 0, seqs = Math.floor(pool / (fixed + perTok * seqLen));
    const k = (W - 4) / mem;
    let s = `<rect x="0" y="18" width="${wPer * k}" height="24" class="w sep"><title>weights: ${fmtBytes(wPer)}</title></rect>`
      + `<rect x="${wPer * k}" y="18" width="${work * k}" height="24" class="a sep"><title>activations and buffers (≈ 5% assumed): ${fmtBytes(work)}</title></rect>`
      + `<rect x="${(wPer + work) * k}" y="18" width="${Math.max(0, pool) * k}" height="24" class="kvbar sep"><title>KV pool: ${fmtBytes(pool)}</title></rect>`
      + `<line x1="${util * mem * k}" x2="${util * mem * k}" y1="10" y2="50" class="memline"/><text x="${Math.min(util * mem * k, W - 60)}" y="8" text-anchor="middle" class="axis">${util} × memory</text>`;
    return `<p class="controls"><label class="nw">weights <select id="kv-w">${[['release', 'published release'], ['bf16', 'BF16'], ['fp8', 'FP8']].map(([v, n]) => `<option value="${v}"${v === wprec ? ' selected' : ''}>${n}</option>`).join('')}</select></label> <label class="nw">KV dtype <select id="kv-d"><option value="bf16"${kvDtype === 'bf16' ? ' selected' : ''}>BF16</option><option value="fp8"${kvDtype === 'fp8' ? ' selected' : ''}>FP8</option></select></label> <label class="nw">memory fraction <select id="kv-u">${[0.8, 0.85, 0.9, 0.95].map((v) => `<option${v === util ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">sequence length <select id="kv-s">${(([4096, 32768, 131072].filter((v) => v <= m.context)).length ? [4096, 32768, 131072].filter((v) => v <= m.context) : [m.context]).map((v) => `<option value="${v}"${v === seqLen ? ' selected' : ''}>${fmtN(v)}</option>`).join('')}</select></label> ${gpnSelect('kv-g')}</p>
      ${figure(s, W, 52, 'GPU memory and KV pool')}
      <p class="status">Per GPU (TP ${tp} × PP ${pp} = ${used} GPUs of ${fmtBytes(mem)}${used < G ? `; ${G - used} of the ${G} GPUs stay unused: ${why}` : ''}): weights ${fmtBytes(wPer)}, KV pool ${fmtBytes(pool)}. At ${fmtBytes(perTok)} per token per GPU${fixed > 0 ? ` plus ${fmtBytes(fixed)} of fixed state per sequence` : ''}, the pool holds <b>${fmtN(tokens)} tokens</b>: ${fmtN(Math.max(0, seqs))} sequences of ${fmtN(seqLen)} at once.</p>
      <p class="caption">Green: weights; orange: activations and buffers (5% of memory assumed); purple: the KV pool. Engines allocate the pool in blocks of positions: vLLM 0.32 uses 16 per block by default and other sizes for some MLA attention backends; for hybrid models such as Qwen3.5 it enlarges the attention block until one block of K/V is at least as large as one linear-attention state. A sequence therefore wastes at most one partly filled block, and sequences can share blocks of a common prefix. K/V heads are split over the GPUs of a node (tensor parallelism; a head is replicated when GPUs outnumber K/V heads) and layers over the nodes (pipeline parallelism).</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    if (seqLen > m.context) seqLen = Math.min(4096, m.context);
    return `
      <p class="lead">During generation each sequence keeps the keys and values of all its earlier positions, so that a new token's attention reads them instead of recomputing them. This KV cache usually takes most of the GPU memory left after the weights, and it sets how many sequences can be served at once and how long they can be. Linear-attention layers keep a fixed-size state instead; sliding-window layers keep only their window.</p>
      <h2>How many tokens fit</h2>${figPool(m, ctx, W)}
      ${measuredTable('kv-cache', m)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>KV bytes per token per GPU = (full-attention layers) × 2 × ⌈kv_heads / TP⌉ × head_dim × bytes per value; MLA: layers × 576 × bytes, the same on every TP rank.</li>
        <li>Tokens that fit = (memory fraction × GPU memory − weights − workspace) / KV bytes per token per GPU.</li>
        <li>FP8 KV cache halves the bytes; attention then reads 8-bit keys and values with a scale.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">PagedAttention (vLLM)</a> · ${configLink(m)}.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#kv-w').addEventListener('change', (e) => { wprec = e.target.value; rerender(); });
    root.querySelector('#kv-d').addEventListener('change', (e) => { kvDtype = e.target.value; rerender(); });
    root.querySelector('#kv-u').addEventListener('change', (e) => { util = Number(e.target.value); rerender(); });
    root.querySelector('#kv-s').addEventListener('change', (e) => { seqLen = Number(e.target.value); rerender(); });
    bindGpn(root, 'kv-g', rerender);
  }

  (window.CHAPTERS = window.CHAPTERS || {})['kv-cache'] = { render, mount };
})();
