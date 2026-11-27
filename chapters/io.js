// Chapter: Input and output.
// Figures: (1) inputs and next-token targets for a window of example tokens, with each position's
// probability for its target and the loss; (2) shapes and bytes for the selected model at a chosen
// window length; (3) context lengths of all models.
(function () {
  const TOKENS = ['The', ' cat', ' sat', ' on', ' the', ' mat', ' and', ' slept', '.'];
  const P_TARGET = [0.004, 0.11, 0.33, 0.52, 0.21, 0.64, 0.07, 0.19];   // illustrative p(target) at positions 1–8
  const NEXT = [['.', 0.19], [' on', 0.15], [' there', 0.07], [' for', 0.05], [' in', 0.05]];
  let pos = 7, T = 8192;

  function figWindow(m, W) {
    const n = 8, colW = Math.min(84, Math.floor((W - 8) / n)), y1 = 18, y2 = 92;
    let s = `<text x="0" y="10" class="axis">input (positions 1–8)</text><text x="0" y="${y2 - 8}" class="axis">target: the next token</text>`;
    for (let i = 0; i < n; i++) {
      const x = i * colW + colW / 2, on = i === pos;
      s += `<g class="pos${on ? ' on' : ''}" data-pos="${i}"><rect x="${i * colW}" y="${y1}" width="${colW}" height="${y2 + 24 - y1}" class="hit"/>`
        + `<text x="${x}" y="${y1 + 18}" text-anchor="middle" class="tok">${showTok(TOKENS[i])}</text>`
        + `<line x1="${x}" x2="${x}" y1="${y1 + 26}" y2="${y2 - 22}" class="${on ? 'arrow-on' : 'arrow'}"/>`
        + `<text x="${x}" y="${y2 + 12}" text-anchor="middle" class="tok">${showTok(TOKENS[i + 1])}</text></g>`;
    }
    const p = P_TARGET[pos], loss = -Math.log(p), mean = P_TARGET.reduce((t, q) => t - Math.log(q), 0) / n;
    const bars = NEXT.map(([t, q]) => `<div class="pbar"><span class="mono">${showTok(t)}</span><span class="track"><span style="width:${q * 100 / 0.25}%"></span></span><span class="mono">${q.toFixed(2)}</span></div>`).join('');
    return `${figure(s, Math.min(W, n * colW), y2 + 26, 'inputs and targets')}
      <p class="status">Position ${pos + 1} reads <code>${showTok(TOKENS[pos])}</code> and everything before it, and must predict <code>${showTok(TOKENS[pos + 1])}</code>. Its probability for that token is ${p} (illustrative), so its loss is −ln ${p} = ${loss.toFixed(2)}. The training loss of this window is the mean over its 8 positions: ${mean.toFixed(2)}.</p>
      ${pos === 7 ? `<div class="pbars"><p class="caption">Probabilities at position 8 for the next token, the five largest of ${fmtN(m.vocab)} (illustrative):</p>${bars}</div>` : ''}
      <p class="caption">Click a position. Each position sees only the tokens up to itself (causal attention), so all 8 predictions come from one forward pass. Probabilities are illustrative; the page does not run the model.</p>`;
  }

  function figShapes(m, W) {
    const P = params(m), V = m.vocab, H = m.hidden;
    const rows = [
      ['token IDs', `T = ${fmtN(T)} integers`, T * 4, 'ink', '32-bit'],
      ['embedding output', `${fmtN(T)} × ${fmtN(H)}`, T * H * 2, 'act', 'BF16; the same size between every pair of layers'],
      ['logits, every position (training)', `${fmtN(T)} × ${fmtN(V)}`, T * V * 4, 'act', 'FP32 for the loss'],
      ['logits, last position (generation)', `1 × ${fmtN(V)}`, V * 4, 'act', 'FP32'],
    ];
    const max = Math.max(...rows.map((r) => r[2])), lw = W < 560 ? 0 : 230, bw = W - lw - (W < 560 ? 0 : 90);
    let s = '';
    rows.forEach(([name, shape, bytes, kind], i) => {
      const y = i * (W < 560 ? 44 : 30) + (W < 560 ? 14 : 2), w = Math.max(1, (Math.log10(bytes) / Math.log10(max)) * bw);
      s += `<text x="0" y="${W < 560 ? y : y + 13}" class="name">${name}</text>`
        + `<rect x="${lw}" y="${W < 560 ? y + 6 : y + 2}" width="${w}" height="14" class="${kind === 'act' ? 'actbar' : 'tokbar'}"><title>${shape}: ${fmtN(bytes)} B</title></rect>`
        + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 13}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${fmtBytes(bytes)}</text>`;
    });
    const steps = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144].filter((x) => x <= m.context);
    const opts = steps.map((x) => `<option value="${x}"${x === T ? ' selected' : ''}>${fmtN(x)}</option>`).join('');
    return `<p class="controls">Window length T <select id="io-T">${opts}</select> tokens (${esc(m.name)} allows up to ${fmtN(m.context)})</p>
      ${figure(s, W, rows.length * (W < 560 ? 44 : 30) + 4, 'shapes and bytes')}
      <table class="data"><thead><tr><th>tensor</th><th>shape</th><th class="r">bytes</th><th>note</th></tr></thead><tbody>${rows.map(([name, shape, bytes, , note]) => `<tr><td>${name}</td><td class="mono">${shape}</td><td class="r">${fmtBytes(bytes)}</td><td>${note}</td></tr>`).join('')}</tbody></table>
      <p class="caption">One sequence of T tokens of ${esc(m.name)} (log-scaled bars; orange: activations). The logits of every position, kept in FP32 for the loss, are the largest single tensor at the output: training frameworks compute the loss in chunks or fuse it with the output layer to avoid storing all of them. Generation needs only the last position's ${fmtN(V)} logits.</p>`;
  }

  function figContexts(m, W) {
    const models = window.MODELS.slice().sort((a, b) => a.context - b.context);
    const narrow = W < 560, lw = narrow ? 0 : 168, rowH = narrow ? 40 : 28, x0 = lw, x1 = W - (narrow ? 0 : 110), sx = logScale(512, 300000, x0, x1);
    let s = '';
    models.forEach((mm, i) => {
      const y = i * rowH + (narrow ? 16 : 4), on = mm.id === m.id;
      s += `<g class="row${on ? ' on' : ''}" data-model="${mm.id}"><rect x="0" y="${y - (narrow ? 14 : 2)}" width="${W}" height="${rowH - 2}" class="hit"/>`
        + `<text x="0" y="${narrow ? y - 2 : y + 14}" class="name">${esc(mm.name)}</text>`
        + `<rect x="${x0}" y="${y + 2}" width="${sx(mm.context) - x0}" height="16" class="tokbar"/>`
        + `<text x="${narrow ? W : x1 + 10}" y="${narrow ? y - 2 : y + 14}"${narrow ? ' text-anchor="end"' : ''} class="val">${fmtN(mm.context)}</text></g>`;
    });
    return `${figure(s, W, models.length * rowH + (narrow ? 20 : 8), 'context lengths')}
      <p class="caption">Maximum window length in tokens (max_position_embeddings in config.json, inside text_config for Qwen3-VL and Qwen3.5; n_positions for GPT-2; log scale). How the models reach these lengths is in Long context.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    if (T > m.context) T = Math.min(8192, m.context);
    return `
      <p class="lead">The network reads a window of up to T token IDs. For every position it outputs V numbers, the logits, one per vocabulary entry; a softmax turns them into probabilities for the token that follows that position. Training compares each position's probabilities with the token that actually follows, so one window of T tokens gives T predictions. Generation uses only the last position's probabilities.</p>
      <p><b>${esc(m.name)}</b> reads windows of up to ${fmtN(m.context)} tokens and outputs ${fmtN(m.vocab)} logits per position. Inside the network each position is a vector of H = ${fmtN(m.hidden)} values.</p>
      <h2>One window, eight predictions</h2>${figWindow(m, W)}
      <h2>Shapes and bytes</h2>${figShapes(m, W)}
      <h2>Context length of each model</h2>${figContexts(m, W)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Probabilities at position t: p<sub>t</sub> = softmax(z<sub>t</sub>), z<sub>t</sub> = h<sub>t</sub>·W<sub>out</sub> ∈ ℝ<sup>V</sup>, W<sub>out</sub> ∈ ℝ<sup>${fmtN(m.hidden)} × ${fmtN(m.vocab)}</sup>.</li>
        <li>Loss of a window: L = −(1/T) Σ<sub>t</sub> ln p<sub>t</sub>[x<sub>t+1</sub>], the cross-entropy between the predictions and the tokens that follow.</li>
        <li>Perplexity e<sup>L</sup>: a uniform distribution over e<sup>L</sup> tokens has loss L. A uniform distribution over all ${fmtN(m.vocab)} entries has L = ln ${fmtN(m.vocab)} = ${Math.log(m.vocab).toFixed(2)}.</li>
      </ul></details>
      <p class="sources">Sources: ${configLink(m)}.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelectorAll('[data-pos]').forEach((g) => g.addEventListener('click', () => { pos = Number(g.dataset.pos); rerender(); }));
    root.querySelector('#io-T').addEventListener('change', (e) => { T = Number(e.target.value); rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).io = { render, mount };
})();
