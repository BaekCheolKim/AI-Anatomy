// Chapter: Serving many requests.
// Figures: (1) a decode step's bytes and FLOPs per generated token against batch size; with GPU speeds
// entered, tokens/s and time per output token; (2) continuous batching on a timeline; (3) measured.
(function () {
  let ctxLen = 4096;
  const BATCH = [1, 4, 16, 64, 256, 1024];

  function figBatch(m, ctx, W) {
    const G = ctx.gpus, rows = BATCH.map((B) => { const st = inferenceStep(m, { mode: 'decode', B, c: ctxLen }); return { B, st, t: stepTime(st, G, hwPeak(), hwBw()) }; });
    const maxB = Math.max(...rows.map((r) => r.st.bytes / r.B)), lw = 90, bw = W - lw - 170, rowH = 24;
    let s = '';
    rows.forEach(({ B, st }, i) => {
      const y = i * rowH + 2, perTok = st.bytes / B;
      s += `<text x="0" y="${y + 14}" class="name">${B} seq.</text><rect x="${lw}" y="${y + 3}" width="${Math.max(1, (perTok / maxB) * bw)}" height="14" class="w"/>`
        + `<text x="${lw + bw + 8}" y="${y + 14}" class="val">${fmtBytes(perTok)} read per token</text>`;
    });
    const times = rows[0].t ? `<table class="data"><thead><tr><th>sequences</th><th class="r">step time ≥</th><th class="r">time per output token</th><th class="r">tokens/s (instance)</th><th>limited by</th></tr></thead><tbody>${rows.map(({ B, t }) => `<tr><td>${B}</td><td class="r">${fmtTime(t.t)}</td><td class="r">${fmtTime(t.t)}</td><td class="r">${fmtN(B / t.t)}</td><td>${t.bound}</td></tr>`).join('')}</tbody></table>` : '';
    return `${hwInputs()}${figure(s, W, rows.length * rowH + 4, 'bytes per generated token')}${times}
      <p class="caption">${esc(m.name)}, decode at ${fmtN(ctxLen)} positions on ${G} GPU${G > 1 ? 's' : ''}: bytes read from memory per generated token. Larger batches share each weight read among more tokens, until the KV cache read (which grows with every sequence) and compute dominate. With GPU speeds entered, the table gives lower bounds: each sequence gets one token per step, so time per output token equals the step time and grows with the batch, while tokens per second for the whole instance grows too.</p>`;
  }

  function figTimeline(W) {
    const reqs = [[0, 3, 9], [1, 2, 5], [2, 4, 12], [5, 2, 4], [7, 3, 6], [9, 1, 8]];   // [arrival step, prefill steps, decode steps]
    const steps = 22, cw = Math.max(10, Math.floor((W - 60) / steps)), rowH = 18;
    let s = '';
    reqs.forEach(([a, p, d], i) => {
      const y = i * rowH;
      s += `<text x="0" y="${y + 12}" class="axis">req ${i + 1}</text>`;
      for (let t = 0; t < p; t++) s += `<rect x="${60 + (a + t) * cw}" y="${y + 2}" width="${cw - 1}" height="12" class="seg0"><title>prefill chunk</title></rect>`;
      for (let t = 0; t < d; t++) s += `<rect x="${60 + (a + p + t) * cw}" y="${y + 2}" width="${cw - 1}" height="12" class="a"><title>decode step</title></rect>`;
    });
    for (let t = 0; t <= steps; t += 5) s += `<text x="${60 + t * cw}" y="${reqs.length * rowH + 14}" class="axis">step ${t}</text>`;
    return `${figure(s, W, reqs.length * rowH + 18, 'continuous batching')}
      <p class="caption">Continuous batching (illustrative): requests arrive at different steps; dark cells are prefill chunks, orange cells decode steps. Every engine step runs all active cells in one batch. Time to first token = waiting + prefill; time per output token = the duration of the steps the request takes part in.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    if (ctxLen > m.context) ctxLen = m.context;
    return `
      <p class="lead">A served model handles many requests at once. Each decode step reads the weights once for all sequences in the batch, so a larger batch produces more tokens per second for the whole instance; but every sequence gets only one token per step, and steps take longer as the batch and its KV cache grow. Serving balances throughput (tokens per second per GPU) against latency (time to first token, time per output token).</p>
      <p class="controls">Context per sequence <select id="sv-c">${[1024, 4096, 16384, 65536].filter((c) => c <= m.context).map((c) => `<option value="${c}"${c === ctxLen ? ' selected' : ''}>${fmtN(c)}</option>`).join('')}</select></p>
      <h2>Batch size</h2>${figBatch(m, ctx, W)}
      <h2>Continuous batching</h2>${figTimeline(W)}
      ${measuredTable('serving', m)}
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">vLLM</a> · <a href="https://arxiv.org/abs/2211.05102" target="_blank" rel="noreferrer">Efficiently Scaling Transformer Inference</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#sv-c').addEventListener('change', (e) => { ctxLen = Number(e.target.value); rerender(); });
    bindHw(root, rerender);
  }

  (window.CHAPTERS = window.CHAPTERS || {}).serving = { render, mount };
})();
