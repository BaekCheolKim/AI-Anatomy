// Chapter: Long context.
// Figures: (1) rotary-embedding wavelengths per dimension pair, before and after the model's
// context-extension scaling; (2) cost of a longer context: prefill FLOPs, attention share, cache
// per sequence, FLOPs per generated token.
(function () {
  const rotaryDims = (m) => (m.attention.kind === 'MLA' ? m.attention.ropeDim : m.attention.headDim * (m.ropeFraction || 1));

  // Frequency of dimension pair i after scaling (radians per position).
  function scaled(m, f) {
    const r = m.ropeScaling;
    if (!r) return f;
    const lam = (2 * Math.PI) / f;
    if (r.type === 'llama3') {
      const lowW = r.original / r.lowFreqFactor, highW = r.original / r.highFreqFactor;
      if (lam < highW) return f;
      if (lam > lowW) return f / r.factor;
      const smooth = (r.original / lam - r.lowFreqFactor) / (r.highFreqFactor - r.lowFreqFactor);
      return (1 - smooth) * (f / r.factor) + smooth * f;
    }
    const rot = r.original / lam, g = rot < r.betaSlow ? 0 : rot > r.betaFast ? 1 : (rot - r.betaSlow) / (r.betaFast - r.betaSlow);
    return (1 - g) * (f / r.factor) + g * f;   // YaRN: interpolate slow pairs, keep fast ones, ramp in between
  }

  function figWavelengths(m, W) {
    const d = rotaryDims(m), n = d / 2, H = 260, padL = 56, padB = 30, padR = 12;
    const sx = (i) => padL + (i / Math.max(1, n - 1)) * (W - padL - padR), sy = logScale(1, 1e9, H - padB, 8);
    let s = '';
    [1, 1e2, 1e4, 1e6, 1e8].forEach((v) => { s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(v)}" y2="${sy(v)}" class="grid"/><text x="${padL - 6}" y="${sy(v) + 4}" text-anchor="end" class="axis">${v >= 1e6 ? fmtParams(v) : fmtN(v)}</text>`; });
    s += `<text x="${padL}" y="${H - 8}" class="axis">dimension pair 0</text><text x="${W - padR}" y="${H - 8}" text-anchor="end" class="axis">pair ${n - 1}</text>`;
    const lines = [[m.context, `context ${fmtN(m.context)}`]];
    if (m.ropeScaling) lines.push([m.ropeScaling.original, `pre-training length ${fmtN(m.ropeScaling.original)}`]);
    lines.forEach(([v, label]) => { s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(v)}" y2="${sy(v)}" class="memline"/><text x="${W - padR}" y="${sy(v) - 5}" text-anchor="end" class="axis">${label}</text>`; });
    const base = range(n).map((i) => m.ropeTheta ** (-(2 * i) / d));
    const path = (fs) => fs.map((f, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(Math.min(1e9, (2 * Math.PI) / f)).toFixed(1)}`).join('');
    s += `<path d="${path(base)}" class="${m.ropeScaling ? 'line-base' : 'line-ink'}"/>`;
    if (m.ropeScaling) s += `<path d="${path(base.map((f) => scaled(m, f)))}" class="line-ink"/>`;
    return `${figure(s, W, H, 'rotary wavelengths')}
      <p class="caption">Wavelength in positions of each of the ${n} rotated dimension pairs (log scale): the distance after which a pair's rotation repeats. ${m.ropeScaling ? 'Dashed line: base frequencies; solid line: after scaling. ' : ''}Pairs whose wavelength exceeds the pre-training length never completed a rotation in pre-training; scaling stretches those pairs, so positions up to the new context map to angles the model has seen.</p>`;
  }

  function figCost(m, W) {
    const lens = [4096, 32768, 131072, m.context].filter((t, i, a) => t <= m.context && a.indexOf(t) === i);
    const rows = lens.map((T) => {
      const f = forwardFlopsPerToken(m, T, 'causal'), dec = forwardFlopsPerToken(m, T, 'decode'), c = cacheBytes(m, T);
      return `<tr><td>${fmtN(T)}</td><td class="r">${fmtFlops(f.total * T)}</td><td class="r">${fmtPct(f.attention, f.total)}</td><td class="r">${fmtBytes(c.total)}</td><td class="r">${fmtFlops(dec.total)}</td></tr>`;
    }).join('');
    return `<table class="data"><thead><tr><th>positions</th><th class="r">prefill FLOPs</th><th class="r">attention share</th><th class="r">cache per sequence</th><th class="r">FLOPs per next token</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="caption">${esc(m.name)}, one sequence. Prefill (reading the whole prompt) grows with T for the weights and with T² for attention; the cache grows with T; each generated token reads the whole cache. Linear-attention and sliding-window layers do not grow with T.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, r = m.ropeScaling;
    let pos;
    if (m.position === 'learned') pos = `<b>${esc(m.name)}</b> uses ${fmtN(m.context)} learned position vectors, one per position, added to the token embeddings. It has no vector for position ${fmtN(m.context + 1)}, so its context cannot be extended without training new position vectors.`;
    else {
      pos = `<b>${esc(m.name)}</b> rotates ${rotaryDims(m)} dimensions of each query and key head${m.attention.kind === 'MLA' ? ' (the decoupled RoPE part of MLA; the other dimensions carry no position)' : m.ropeFraction ? ` (${m.ropeFraction * 100}% of the head; the rest carries no position)` : ''}, base θ = ${fmtN(m.ropeTheta)}.`;
      if (r) pos += ` Its frequencies are rescaled with ${r.type === 'llama3' ? `Llama 3 scaling (factor ${r.factor}; pairs with wavelengths above ${fmtN(r.original / r.lowFreqFactor)} positions divided by ${r.factor}, below ${fmtN(r.original / r.highFreqFactor)} kept, smooth in between)` : `YaRN (factor ${r.factor} from ${fmtN(r.original)} positions; slow pairs interpolated, fast pairs kept, attention logits rescaled)`} to reach ${fmtN(m.context)} positions.`;
      else pos += ` Its configuration has no rescaling: ${fmtN(m.context)} positions is the length it was trained for.`;
      if (m.mrope) pos += ' The rotated dimensions are split into three groups for time, height and width (M-RoPE), so image and video tokens get 3-D positions.';
      if (m.longContext) pos += ` ${esc(m.longContext.text)}`;
    }
    return `
      <p class="lead">A model is trained up to a maximum context length. Most current models encode position by rotating query and key vectors by an angle proportional to the position (rotary embedding, RoPE); each pair of dimensions rotates at its own frequency. Pre-training runs at a short length because attention cost grows with the square of the length. Longer contexts are added afterwards: the slow frequencies are rescaled and training continues on long documents.</p>
      <p>${pos}</p>
      ${m.position === 'rope' ? `<h2>Rotation wavelengths</h2>${figWavelengths(m, W)}` : ''}
      <h2>What a longer context costs</h2>${figCost(m, W)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>RoPE: pair i of a head with d rotated dimensions turns by angle p·θ<sub>i</sub> at position p, θ<sub>i</sub> = base<sup>−2i/d</sup>; wavelength 2π/θ<sub>i</sub>. The score q·k then depends only on the distance between the two positions.</li>
        <li>Position interpolation divides every θ<sub>i</sub> by the extension factor s. Llama 3 scaling and YaRN divide only the slow pairs and keep the fast ones, which encode short distances.</li>
        <li>YaRN also multiplies the attention logits by about (0.1·ln s + 1)² to keep the softmax distribution's sharpness at longer lengths.</li>
        <li>Prefill FLOPs ≈ 2·N·T + attention; attention for causal softmax layers ≈ n<sub>heads</sub>·T²·(d<sub>qk</sub> + d<sub>v</sub>) per layer.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noreferrer">RoPE</a> · <a href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">YaRN</a> · <a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noreferrer">Llama 3 paper</a> · <a href="${esc(m.source)}" target="_blank" rel="noreferrer">${esc(m.name)} config.json</a>.</p>`;
  }

  (window.CHAPTERS = window.CHAPTERS || {})['long-context'] = { render };
})();
