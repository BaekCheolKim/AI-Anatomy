// Chapter: Attention variants.
// Figures: (1) values each model stores per position per layer, and per token over all layers;
// (2) cache per sequence against context length for every model, with the reader's GPU memory;
// (3) which positions each variant reads (causal, sliding window, linear state).
(function () {
  const VARIANT = {
    MHA: 'Multi-head attention (MHA): every query head has its own key and value head.',
    GQA: 'Grouped-query attention (GQA): several query heads share one key and value head, so fewer K/V values are stored per position.',
    MLA: 'Multi-head latent attention (MLA): keys and values are reconstructed from one compressed latent vector per position; only the latent (and one shared RoPE key) is stored.',
  };

  function describe(m) {
    const a = m.attention, t = layerTypes(m), full = t.filter((x) => x === 'full').length, perPos = cacheValues(m, 1).perPos;
    const parts = [`<b>${esc(m.name)}</b>: ${VARIANT[a.kind]}`];
    if (a.kind === 'MLA') parts.push(`It stores ${a.kvRank} + ${a.ropeDim} = ${fmtN(perPos)} values per position per layer, Caching the keys and values it reconstructs would take ${a.heads} × (${a.nopeDim + a.ropeDim} + ${a.vDim}) = ${fmtN(a.heads * (a.nopeDim + a.ropeDim + a.vDim))} values; MHA with ${a.heads} heads of ${a.vDim}, the DeepSeek-V2 paper's comparison, would store ${fmtN(2 * a.heads * a.vDim)}.`);
    else parts.push(`${a.heads} query heads, ${a.kvHeads} K/V heads of ${a.headDim}: ${fmtN(perPos)} values per position per layer.`);
    if (t.includes('sliding')) parts.push(`${t.filter((x) => x === 'sliding').length} of its ${m.layers} layers attend only to the last ${a.window} positions, so their cache stops growing at ${a.window} positions. `);
    if (a.sinks) parts.push('In every layer, each attention head has a learned sink logit s that adds e<sup>s</sup> to the softmax denominator, so its weights over positions sum to less than 1.');
    if (t.includes('linear')) {
      const g = m.linearAttention;
      parts.push(`Only ${full} of its ${m.layers} layers use softmax attention with a growing cache. The other ${m.layers - full} are Gated DeltaNet layers: each keeps a state of ${g.valueHeads} × ${g.keyDim} × ${g.valueDim} values per sequence, updated at every token and the same size at any length.`);
    }
    if (a.outputGate) parts.push('The attention output is multiplied by a sigmoid gate computed from the input.');
    if (a.qkNorm) parts.push('Queries and keys are RMS-normalized per head before the dot product.');
    return `<p>${parts.join(' ')}</p>`;
  }

  // ---- figure 1: values per position per layer, and bytes per token ----
  function figPerPosition(m, W) {
    const models = window.MODELS.slice().sort((a, b) => kvBytesPerToken(b) - kvBytesPerToken(a));
    const narrow = W < 560, lw = narrow ? 0 : 168, rowH = narrow ? 42 : 28, bw = W - lw - (narrow ? 0 : 200);
    const max = Math.max(...models.map((x) => kvBytesPerToken(x)));
    let s = '';
    models.forEach((mm, i) => {
      const y = i * rowH + (narrow ? 16 : 2), on = mm.id === m.id, b = kvBytesPerToken(mm), t = layerTypes(mm);
      const growing = t.filter((x) => x === 'full').length, perPos = cacheValues(mm, 1).perPos;
      const val = `${fmtN(perPos)} values × ${growing} layers = ${fmtBytes(b)}`;
      s += `<g class="row${on ? ' on' : ''}" data-model="${mm.id}"><title>${esc(mm.name)}: ${fmtN(b)} B per token (BF16)</title><rect x="0" y="${y - (narrow ? 14 : 0)}" width="${W}" height="${rowH - 2}" class="hit"/>`
        + `<text x="0" y="${narrow ? y - 2 : y + 15}" class="name">${esc(mm.name)}</text>`
        + `<rect x="${lw}" y="${y + 3}" width="${(b / max) * bw}" height="14" class="kvbar"/>`
        + `<text x="${narrow ? W : lw + bw + 8}" y="${narrow ? y - 2 : y + 15}"${narrow ? ' text-anchor="end"' : ''} class="val">${val}</text></g>`;
    });
    return `${figure(s, W, models.length * rowH + (narrow ? 16 : 0), 'cache bytes per token')}
      <p class="caption">KV-cache bytes added per token in BF16, at contexts longer than any sliding window: values stored per position per layer × layers whose cache grows with the context. Sliding-window layers (gpt-oss) and linear-attention layers (Qwen3.5) are not counted here; they hold a fixed amount per sequence (next figure). Click a row to select that model.</p>`;
  }

  // ---- figure 2: cache per sequence vs context ----
  function figGrowth(m, ctx, W) {
    const H = 260, padL = 52, padB = 26, padR = W < 560 ? 8 : 150, x0 = 1024, x1 = 262144;
    const sx = logScale(x0, x1, padL, W - padR), y0 = 1e7, y1 = 3e11, sy = (v) => padB / 2 + (1 - (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (H - padB - padB / 2);
    let s = '';
    [1e7, 1e8, 1e9, 1e10, 1e11].forEach((v) => { s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(v)}" y2="${sy(v)}" class="grid"/><text x="${padL - 6}" y="${sy(v) + 4}" text-anchor="end" class="axis">${fmtBytes(v)}</text>`; });
    [1024, 8192, 32768, 131072, 262144].forEach((c) => { s += `<text x="${sx(c)}" y="${H - 6}" text-anchor="middle" class="axis">${c >= 1024 ? `${c / 1024}K` : c}</text>`; });
    if (ctx.mem >= y0 && ctx.mem <= y1) s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(ctx.mem)}" y2="${sy(ctx.mem)}" class="memline"/><text x="${padL + 4}" y="${sy(ctx.mem) - 5}" class="axis">GPU memory ${fmtBytes(ctx.mem)}</text>`;
    const pts = [];
    for (let e = 10; e <= 18; e += 0.25) pts.push(2 ** e);
    const wide = padR > 100, labels = [];
    window.MODELS.forEach((mm) => {
      const on = mm.id === m.id, xs = pts.filter((c) => c <= mm.context), last = xs[xs.length - 1];
      const yv = (c) => sy(Math.max(y0, cacheBytes(mm, c).total));
      const d = xs.map((c, i) => `${i ? 'L' : 'M'}${sx(c).toFixed(1)},${yv(c).toFixed(1)}`).join('');
      s += xs.length > 1 ? `<path d="${d}" class="${on ? 'kvline-on' : 'kvline'}"><title>${esc(mm.name)}</title></path>`
        : `<circle cx="${sx(last)}" cy="${yv(last)}" r="${on ? 4 : 3}" class="${on ? 'kvdot-on' : 'kvdot'}"><title>${esc(mm.name)}</title></circle>`;
      labels.push({ x: sx(last), y0: yv(last), y: yv(last), text: mm.short, on });
    });
    if (wide) {
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) if (labels[i].y < labels[i - 1].y + 13) labels[i].y = labels[i - 1].y + 13;
      labels.forEach((l) => {
        s += `<line x1="${l.x + 3}" x2="${W - padR + 6}" y1="${l.y0}" y2="${l.y}" class="leader"/><text x="${W - padR + 10}" y="${l.y + 4}" class="${l.on ? 'name' : 'axis'}">${esc(l.text)}</text>`;
      });
    } else {
      const l = labels.find((x) => x.on);
      s += `<text x="${Math.min(l.x, W - 4)}" y="${l.y0 - 8}" text-anchor="end" class="name">${esc(l.text)}</text>`;
    }
    const c32 = cacheBytes(m, 32768), cmax = cacheBytes(m, m.context);
    return `${figure(s, W, H, 'cache per sequence versus context')}
      <p class="status">${esc(m.name)}: ${fmtBytes(c32.total)} per sequence at 32,768 positions; ${fmtBytes(cmax.total)} at its maximum of ${fmtN(m.context)}${c32.state ? `, of which ${fmtBytes(c32.state)} is the fixed linear-attention state (FP32, per the config)` : ''}. ${cmax.total > ctx.mem ? `One sequence at full length needs more than one GPU of ${fmtBytes(ctx.mem)} for its cache alone.` : `One GPU of ${fmtBytes(ctx.mem)} holds ${Math.floor(ctx.mem / cmax.total)} such full-length sequence caches, before weights.`}</p>
      <p class="caption">Cache per sequence (log–log), whole model: keys, values and convolution inputs in BF16, the Qwen3.5 linear-attention state in FP32 (mamba_ssm_dtype), against context length up to each model's maximum. Flat parts: sliding-window or linear-attention layers, which stop growing. The dashed line is the GPU memory set at the top.</p>`;
  }

  // ---- figure 3: which positions each variant reads ----
  function figMasks(W) {
    const n = 12, cell = Math.min(14, Math.floor((W - 40) / (3 * n + 6))), gap = 1, size = n * (cell + gap), win = 4;
    const panel = (x0, title, read) => {
      let s = `<text x="${x0}" y="12" class="name">${title}</text>`;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const r = read(i, j);
        s += `<rect x="${x0 + j * (cell + gap)}" y="${22 + i * (cell + gap)}" width="${cell}" height="${cell}" class="${r ? 'kvcell' : 'emptycell'}"/>`;
      }
      return s;
    };
    const gapX = Math.max(16, (W - 3 * size) / 2);
    let s = panel(0, 'full (causal)', (i, j) => j <= i) + panel(size + gapX, `sliding window (${win} shown)`, (i, j) => j <= i && j > i - win);
    const lx = 2 * (size + gapX);
    s += `<text x="${lx}" y="12" class="name">linear attention</text>`;
    for (let i = 0; i < n; i++) {
      s += `<rect x="${lx}" y="${22 + i * (cell + gap)}" width="${cell}" height="${cell}" class="kvcell"/>`;
      s += `<rect x="${lx + 2 * (cell + gap)}" y="${22 + i * (cell + gap)}" width="${cell}" height="${cell}" class="emptycell"/>`;
    }
    s += `<text x="${lx + 4 * (cell + gap)}" y="${22 + cell}" class="axis">state S, read at</text><text x="${lx + 4 * (cell + gap)}" y="${22 + cell + 13}" class="axis">every position</text>`;
    return `${figure(s, Math.max(W, 3 * size + 2 * 16), 22 + size + 4, 'attention masks')}
      <p class="caption">Rows: the position being computed; columns: earlier positions whose cached keys and values it reads (filled). Full attention reads every earlier position. A sliding window reads the last w (4 here; 128 in gpt-oss). Linear attention reads no cached positions: it reads one state that summarizes all earlier positions.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    return `
      <p class="lead">Attention lets each position read earlier positions. The current position's query vector is compared with the key vectors of earlier positions; the softmax of these scores weights their value vectors. During generation the keys and values of earlier positions are kept in GPU memory, the KV cache, so they are computed only once. The variants differ in what they store per position and in how many positions they read.</p>
      ${describe(m)}
      <h2>Bytes per token in the cache</h2>${figPerPosition(m, W)}
      <h2>Cache per sequence and context length</h2>${figGrowth(m, ctx, W)}
      <h2>Which positions are read</h2>${figMasks(W)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Softmax attention, one head: o<sub>t</sub> = Σ<sub>j ≤ t</sub> softmax<sub>j</sub>(q<sub>t</sub>·k<sub>j</sub> / √d)·v<sub>j</sub>. With a sink logit s: the denominator gets an extra e<sup>s</sup> term.</li>
        <li>GQA: query head i uses K/V head ⌊i·n<sub>kv</sub> / n<sub>q</sub>⌋. Cache per position per layer: 2·n<sub>kv</sub>·d values.</li>
        <li>MLA: c<sub>KV</sub> = RMSNorm(x·W<sub>DKV</sub>) (512 values) and k<sub>R</sub> = RoPE(x·W<sub>KR</sub>) (64 values) are cached; k = [c<sub>KV</sub>·W<sub>UK</sub>, k<sub>R</sub>], v = c<sub>KV</sub>·W<sub>UV</sub>. In decode, q·k = (q<sub>C</sub>·W<sub>UK</sub>ᵀ)·c<sub>KV</sub> + q<sub>R</sub>·k<sub>R</sub>, so scores are computed against the latent directly.</li>
        <li>Gated DeltaNet, per value head, with q<sub>t</sub> and k<sub>t</sub> L2-normalized per head and q<sub>t</sub> scaled by 1/√d<sub>k</sub>: S<sub>t</sub> = α<sub>t</sub>·S<sub>t−1</sub>·(I − β<sub>t</sub> k<sub>t</sub>k<sub>t</sub>ᵀ) + β<sub>t</sub> v<sub>t</sub>k<sub>t</sub>ᵀ, o<sub>t</sub> = S<sub>t</sub> q<sub>t</sub>, with decay α<sub>t</sub> = exp(−exp(A_log)·softplus(a<sub>t</sub> + dt_bias)) ∈ (0, 1) and write strength β<sub>t</sub> = σ(b<sub>t</sub>) ∈ (0, 1), where a<sub>t</sub> and b<sub>t</sub> are linear projections of the input.</li>
        <li>Cache bytes per sequence = Σ<sub>layers</sub> values per position × positions kept × bytes per value, plus fixed states.</li>
      </ul></details>
      <p class="sources">Sources: ${configLink(m)} · <a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noreferrer">GQA</a> · <a href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noreferrer">MLA (DeepSeek-V2)</a> · <a href="https://arxiv.org/abs/2412.06464" target="_blank" rel="noreferrer">Gated DeltaNet</a> · <a href="https://arxiv.org/abs/2508.10925" target="_blank" rel="noreferrer">gpt-oss model card (OpenAI, 2025), section 2</a>.</p>`;
  }

  (window.CHAPTERS = window.CHAPTERS || {}).attention = { render };
})();
