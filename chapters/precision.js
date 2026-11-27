// Chapter: Precision and quantization.
// Figures: (1) bytes per weight of common formats, scales included; (2) the selected model's
// weights in each format, with the tensors kept in BF16; (3) measured next to computed.
(function () {
  // [name, bytes per weight including scales, how the scale is stored]
  const FORMATS = [
    ['FP32', 4, 'IEEE single precision; training master weights'],
    ['BF16', 2, '8 exponent bits, 7 mantissa bits; the usual release format'],
    ['FP8 (E4M3), 128 × 128 blocks', 1 + 4 / 16384, 'one FP32 scale per block of 16,384 weights (DeepSeek-V3, Qwen FP8 releases)'],
    ['INT8, per channel', 1 + 2 / 4096, 'one BF16 scale per output channel of 4,096 weights'],
    ['INT4 (AWQ, GPTQ), groups of 128', 0.5 + 2.5 / 128, 'one FP16 scale and one 4-bit zero point (8 packed per int32) per 128 weights'],
    ['MXFP4', 0.53125, '4-bit values with one 8-bit power-of-two scale per 32 (gpt-oss experts)'],
    ['NVFP4', 0.5625, '4-bit values with one FP8 scale per 16, plus one FP32 scale per tensor'],
    ['GGUF Q4_K_M', 4.85 / 8, 'llama.cpp Q4_K blocks (4-bit values; 6-bit scale and minimum per 32 weights; FP16 scale and minimum per 256: 4.5 bits), with Q6_K (8-bit scale per 16 weights: 6.5625 bits) for half of the attn_v and ffn_down tensors; ≈ 4.85 bits per weight over the linear layers'],
  ];
  let fmt = 2;

  function figFormats(m, W) {
    const c = weightClasses(m), keep = c.embed + c.small + c.vision, quant = c.experts + c.linear;
    const lw = W < 560 ? 0 : 230, bw = W - lw - (W < 560 ? 0 : 90), rowH = W < 560 ? 40 : 26, max = (keep + quant) * 4;
    let s = '';
    FORMATS.forEach(([name, b], i) => {
      const y = i * rowH + (W < 560 ? 14 : 2), total = keep * Math.max(2, b) + quant * b, on = i === fmt;
      s += `<g class="row${on ? ' on' : ''}" data-fmt="${i}"><rect x="0" y="${y - (W < 560 ? 14 : 0)}" width="${W}" height="${rowH - 2}" class="hit"/>`
        + `<text x="0" y="${W < 560 ? y : y + 14}" class="name">${name}</text><rect x="${lw}" y="${W < 560 ? y + 4 : y + 3}" width="${(total / max) * bw}" height="14" class="w"/>`
        + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 14}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${fmtBytes(total)}</text></g>`;
    });
    const [name, b, how] = FORMATS[fmt], total = keep * Math.max(2, b) + quant * b;
    return `${figure(s, W, FORMATS.length * rowH + (W < 560 ? 14 : 4), 'weights per format')}
      <p class="status">${esc(m.name)} in ${name}: ${b >= 2 ? `every weight in ${name} (${esc(how)}), ${b} bytes per parameter.` : `${(b * 8).toFixed(2)} bits per quantized weight (${esc(how)}). The ${fmtParams(quant)} parameters of the linear layers and experts take ${fmtBytes(quant * b)}; embedding, output layer, norms, routers${c.vision ? ' and vision encoder' : ''} (${fmtParams(keep)}) stay in 16-bit floating point (BF16; FP16 in AWQ and GPTQ checkpoints): ${fmtBytes(keep * 2)}.`} Total <b>${fmtBytes(total)}</b>: at least ${Math.ceil(total / ctx_mem)} GPU${Math.ceil(total / ctx_mem) > 1 ? 's' : ''} of ${fmtBytes(ctx_mem)}.</p>
      <p class="caption">Weights of ${esc(m.name)} in each format. Click a row. Low-bit formats store one scale per block of weights, so their bits per weight are above the nominal 4 or 8. Quality loss depends on the model and the method. llama.cpp's Q4_K_M also stores the token embedding in Q4_K and the output layer in Q6_K; the total here keeps them in 16-bit.</p>`;
  }
  let ctx_mem = 80e9;

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    ctx_mem = ctx.mem;
    return `
      <p class="lead">Weights are stored with fewer bits to make a model smaller and faster to read from memory. A quantized format keeps low-bit values plus a scale for each block of weights; each matrix multiplication multiplies the stored values directly (on GPUs with FP8 or FP4 arithmetic) or after converting them back to BF16. Embedding tables, the output layer, norms and routers usually stay in BF16, as in the FP8 releases of DeepSeek-V3 and Qwen. Norms and routers are small; the embedding and output layer are ${fmtPct(params(m).embed + params(m).head, params(m).total)} of ${esc(m.name)}'s parameters.</p>
      <p><b>${esc(m.name)}</b> is published with ${Object.entries(releaseDtypes(m)).filter(([k]) => weightClasses(m)[k] > 0).map(([k, v]) => `${CLASS_NAME[k]} in ${DTYPE_NAME[v]}`).join(', ')}: ${fmtBytes(weightBytes(m).total)}.</p>
      <h2>Bytes per weight</h2>${figFormats(m, W)}
      ${measuredTable('precision', m)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Block quantization: w ≈ s·q, with q the stored low-bit value and s the block's scale (and an offset for formats with a zero point).</li>
        <li>Bits per weight = value bits + scale bits / block size: MXFP4 4 + 8/32 = 4.25; NVFP4 4 + 8/16 = 4.5.</li>
        <li>Decode reads every weight the step uses (all weights of a dense model; the shared and selected experts of an MoE model), so fewer bytes per weight shorten the memory time of each step about proportionally.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2209.05433" target="_blank" rel="noreferrer">FP8 formats</a> · <a href="https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf" target="_blank" rel="noreferrer">OCP microscaling (MX) formats</a> · <a href="https://arxiv.org/abs/2306.00978" target="_blank" rel="noreferrer">AWQ</a> · <a href="https://arxiv.org/abs/2210.17323" target="_blank" rel="noreferrer">GPTQ</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelectorAll('[data-fmt]').forEach((g) => g.addEventListener('click', () => { fmt = Number(g.dataset.fmt); rerender(); }));
  }

  (window.CHAPTERS = window.CHAPTERS || {}).precision = { render, mount };
})();
