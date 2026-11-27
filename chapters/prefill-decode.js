// Chapter: Prefill and decode.
// Figures: (1) FLOPs per byte read for prefill and for decode at several batch sizes; (2) one step
// of each: FLOPs, bytes, and, if the reader gives GPU speeds, a time lower bound and tokens/s.
(function () {
  let ctxLen = 8192;
  const BATCH = [1, 8, 32, 128, 512];

  function figIntensity(m, W) {
    const rows = [['prefill, 1 sequence', inferenceStep(m, { mode: 'prefill', B: 1, T: ctxLen })], ...BATCH.map((B) => [`decode, ${B} sequence${B > 1 ? 's' : ''}`, inferenceStep(m, { mode: 'decode', B, c: ctxLen })])];
    const lw = W < 560 ? 0 : 170, bw = W - lw - (W < 560 ? 0 : 110), rowH = W < 560 ? 40 : 26, sx = logScale(0.5, 20000, 0, bw);
    let s = '';
    rows.forEach(([name, st], i) => {
      const y = i * rowH + (W < 560 ? 14 : 2);
      s += `<text x="0" y="${W < 560 ? y : y + 14}" class="name">${name}</text><rect x="${lw}" y="${W < 560 ? y + 4 : y + 2}" width="${Math.max(1, sx(st.intensity))}" height="16" class="${i ? 'a' : 'seg0'}"/>`
        + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 14}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${st.intensity >= 100 ? fmtN(st.intensity) : st.intensity.toFixed(1)} FLOP/B</text>`;
    });
    const ridge = hwPeak() > 0 && hwBw() > 0 ? hwPeak() / hwBw() : null;
    if (ridge) { const x = lw + sx(ridge); s += `<line x1="${x}" x2="${x}" y1="0" y2="${rows.length * rowH + 6}" class="memline"/><text x="${x + 4}" y="${rows.length * rowH + 16}" class="axis">your GPU: ${fmtN(ridge)} FLOP/B</text>`; }
    return `${figure(s, W, rows.length * rowH + (ridge ? 22 : 8), 'FLOPs per byte')}
      <p class="caption">FLOPs per byte read from memory (log scale) for ${esc(m.name)} at ${fmtN(ctxLen)} positions, release precision. A GPU finishes a step in compute time when this exceeds its peak FLOP/s divided by its memory bandwidth (hundreds of FLOPs per byte on current data-center GPUs), and in memory time otherwise.${ridge ? ' The dashed line is the ratio for the GPU speeds entered below.' : ''}</p>`;
  }

  function stepTable(m, ctx) {
    const G = ctx.gpus, pf = inferenceStep(m, { mode: 'prefill', B: 1, T: ctxLen }), dc = inferenceStep(m, { mode: 'decode', B: 32, c: ctxLen });
    const tp = stepTime(pf, G, hwPeak(), hwBw()), td = stepTime(dc, G, hwPeak(), hwBw());
    const cell = (t, st, label) => (t ? `${fmtTime(t.t)} (${t.bound}-limited), ${fmtN(st.tokens / t.t)} ${label}/s` : '—');
    return `${hwInputs()}
      <table class="data"><thead><tr><th></th><th>prefill: 1 sequence of ${fmtN(ctxLen)} tokens</th><th>decode: 32 sequences, 1 token each</th></tr></thead><tbody>
      <tr><td>FLOPs</td><td>${fmtFlops(pf.flops)}</td><td>${fmtFlops(dc.flops)}</td></tr>
      <tr><td>bytes read or written</td><td>${fmtBytes(pf.bytes)} (weights ${fmtBytes(pf.weightsRead)}, cache written ${fmtBytes(pf.cacheWrite)})</td><td>${fmtBytes(dc.bytes)} (weights ${fmtBytes(dc.weightsRead)}, cache read ${fmtBytes(dc.cacheRead)})</td></tr>
      <tr><td>per GPU (${G} GPU${G > 1 ? 's' : ''})</td><td>${fmtFlops(pf.flops / G)}, ${fmtBytes(pf.bytes / G)}</td><td>${fmtFlops(dc.flops / G)}, ${fmtBytes(dc.bytes / G)}</td></tr>
      <tr><td>time, lower bound</td><td>${cell(tp, pf, 'prompt tokens')}</td><td>${cell(td, dc, 'generated tokens')}</td></tr></tbody></table>
      <p class="caption">Weights and caches are spread evenly over the GPUs set at the top. Times appear when GPU speeds are entered; they assume peak throughput and no overlap losses, so real steps take longer.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    let opts = [2048, 8192, 32768, 131072].filter((c) => c <= m.context);
    if (!opts.length) opts = [m.context];
    if (!opts.includes(ctxLen)) ctxLen = opts[Math.min(1, opts.length - 1)];
    return `
      <p class="lead">Generating a reply has two phases. Prefill processes the whole prompt in one pass: every weight read from memory is used for all prompt tokens, so prefill is limited by arithmetic. Decode then produces one token per sequence per step: every weight is read again for just one token of each sequence, so decode is limited by how fast memory delivers the weights and the KV cache. Batching many sequences into one decode step reuses each weight read for all of them.</p>
      <p class="controls">Context <select id="pd-c">${opts.map((c) => `<option value="${c}"${c === ctxLen ? ' selected' : ''}>${fmtN(c)}</option>`).join('')}</select> positions</p>
      <h2>FLOPs per byte</h2>${figIntensity(m, W)}
      <h2>One step of each</h2>${stepTable(m, ctx)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Decode, B sequences at context c: FLOPs = B·(2·N + attention(c)); bytes = weights read + B·cache(c). With BF16 weights and small B, FLOPs per byte ≈ B: each 2-byte weight does 2 FLOPs per sequence.</li>
        <li>For an MoE model, a decode step reads only the routed experts its B tokens select. If each token picks k of E experts uniformly and independently, the expected number per layer is E·(1 − (1 − k/E)<sup>B</sup>). Actual routing is not uniform, and shared experts are read at every step.</li>
        <li>Multi-head latent attention in decode is computed in absorbed form: 2·(576 + 512) FLOPs per cached position and query head, on 576 cached values. For DeepSeek-V3's 128 query heads on one GPU and a BF16 cache this is 128·1,088/576 ≈ 242 FLOPs per byte, near the compute limit. Splitting the heads over t GPUs divides it by t, because every GPU reads the whole latent.</li>
        <li>Speculative decoding (draft tokens from a small model or an MTP module, verified in one pass) raises FLOPs per byte by checking several tokens per weight read.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2211.05102" target="_blank" rel="noreferrer">Efficiently Scaling Transformer Inference (Pope et al.)</a> · <a href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noreferrer">MLA</a> · <a href="https://arxiv.org/abs/2211.17192" target="_blank" rel="noreferrer">speculative decoding</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#pd-c').addEventListener('change', (e) => { ctxLen = Number(e.target.value); rerender(); });
    bindHw(root, rerender);
  }

  (window.CHAPTERS = window.CHAPTERS || {})['prefill-decode'] = { render, mount };
})();
