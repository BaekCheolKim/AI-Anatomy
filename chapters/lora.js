// Chapter: Fine-tuning: full or LoRA.
// Figures: (1) one weight matrix with a rank-r update; (2) trainable parameters and training memory
// for full fine-tuning, LoRA on BF16 weights and LoRA on 4-bit weights (QLoRA), against GPU memory.
(function () {
  let rank = 16, targets = 'attn';

  // Matrices adapted per layer: [d_in, d_out] for each; attention projections of softmax and linear layers, optionally FFN.
  function adapted(m) {
    const H = m.hidden, a = m.attention, out = [];
    layerTypes(m).forEach((t, i) => {
      if (t === 'linear') { const g = m.linearAttention; out.push([H, 2 * g.keyHeads * g.keyDim + 2 * g.valueHeads * g.valueDim], [g.valueHeads * g.valueDim, H]); }
      else if (a.kind === 'MLA') out.push([H, a.qRank], [a.qRank, a.heads * (a.nopeDim + a.ropeDim)], [H, a.kvRank + a.ropeDim], [a.kvRank, a.heads * (a.nopeDim + a.vDim)], [a.heads * a.vDim, H]);
      else out.push([H, a.heads * a.headDim * (a.outputGate ? 2 : 1)], [H, a.kvHeads * a.headDim], [H, a.kvHeads * a.headDim], [a.heads * a.headDim, H]);
      if (targets === 'all') {
        const f = m.ffn;
        if (!isMoE(m)) out.push(...(f.gated ? [[H, f.inter], [H, f.inter], [f.inter, H]] : [[H, f.inter], [f.inter, H]]));
        else if (i < f.denseLayers) out.push([H, f.denseInter], [H, f.denseInter], [f.denseInter, H]);
        else for (let e = 0; e < f.experts + f.shared; e++) out.push([H, f.expertInter], [H, f.expertInter], [f.expertInter, H]);
      }
    });
    return out;
  }

  function figMatrix(m, W) {
    const H = m.hidden, a = m.attention, din = a.kind === 'MLA' ? a.heads * a.vDim : a.heads * a.headDim;   // attention output: d_in = heads × head dim, d_out = H
    const s0 = Math.min(180, (W - 120) / 2.2), r = Math.max(3, (rank / H) * s0 * 4);
    let s = `<rect x="0" y="0" width="${s0}" height="${s0}" class="w-frozen"/><text x="${s0 / 2}" y="${s0 / 2 + 4}" text-anchor="middle" class="name">W (frozen)</text><text x="${s0 / 2}" y="${s0 + 16}" text-anchor="middle" class="axis">${fmtN(H)} × ${fmtN(din)}</text>`;
    const x1 = s0 + 30;
    s += `<text x="${x1 - 16}" y="${s0 / 2 + 5}" class="name">+</text>`;
    s += `<rect x="${x1}" y="0" width="${r}" height="${s0}" class="w"/><text x="${x1 + r / 2}" y="${s0 + 16}" text-anchor="middle" class="axis">B: ${fmtN(H)} × ${rank}</text>`;
    s += `<rect x="${x1 + r + 12}" y="0" width="${s0}" height="${r}" class="w"/><text x="${x1 + r + 12 + s0 / 2}" y="${r + 16}" text-anchor="middle" class="axis">A: ${rank} × ${fmtN(din)}</text>`;
    return `${figure(s, Math.min(W, x1 + r + 12 + s0 + 4), s0 + 24, 'LoRA update')}
      <p class="caption">The attention output projection of ${esc(m.name)}, W<sub>0</sub> ∈ ℝ<sup>d<sub>out</sub> × d<sub>in</sub></sup> with d<sub>out</sub> = ${fmtN(H)} (the model width) and d<sub>in</sub> = ${fmtN(din)} (heads × head dimension). LoRA keeps W<sub>0</sub> fixed and trains B·A instead: ${fmtN(rank * (H + din))} parameters (r·(d<sub>in</sub> + d<sub>out</sub>)) in place of ${fmtN(H * din)}. The thin sides of B and A are drawn 4× wider than to scale.</p>`;
  }

  function figMemory(m, ctx, W) {
    const P = params(m), loraP = adapted(m).reduce((s, [i, o]) => s + rank * (i + o), 0);
    const nf4 = (4 + 8 / 64 + 32 / (64 * 256)) / 8, keep = P.embed + P.head;   // NF4 with double-quantized scales: 4.127 bits; embedding and output layer stay BF16
    const options = [
      ['Full fine-tuning', P.total * 18, `every parameter trained: BF16 weights, FP32 gradients, FP32 master weights, FP32 AdamW moments, 18 B per parameter`],
      ['LoRA, BF16 base', P.total * 2 + loraP * 18, `frozen BF16 weights (2 B per parameter) + 18 B per LoRA parameter`],
      ['QLoRA, 4-bit base', (P.total - keep) * nf4 + keep * 2 + loraP * 18, `frozen linear layers in 4-bit NF4 with double-quantized scales (4.127 bits ≈ 0.516 B per parameter), embedding and output layer in BF16, + 18 B per LoRA parameter`],
    ];
    const max = Math.max(...options.map((o) => o[1]), ctx.mem), lw = W < 560 ? 0 : 150, bw = W - lw - (W < 560 ? 0 : 90), rowH = W < 560 ? 40 : 28;
    let s = '';
    options.forEach(([name, bytes], i) => {
      const y = i * rowH + (W < 560 ? 14 : 2);
      s += `<text x="0" y="${W < 560 ? y : y + 14}" class="name">${name}</text><rect x="${lw}" y="${W < 560 ? y + 4 : y + 2}" width="${Math.max(1, (bytes / max) * bw)}" height="16" class="${i ? 'w' : 'g'}"/>`
        + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 14}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${fmtBytes(bytes)}</text>`;
    });
    const mx = lw + (ctx.mem / max) * bw;
    s += `<line x1="${mx}" x2="${mx}" y1="0" y2="${options.length * rowH + 4}" class="memline"/>`;
    return `<p class="controls"><label class="nw">rank r <select id="lo-r">${[4, 8, 16, 32, 64, 128, 256].map((v) => `<option${v === rank ? ' selected' : ''}>${v}</option>`).join('')}</select></label> &nbsp; <label class="nw">adapt <select id="lo-t"><option value="attn"${targets === 'attn' ? ' selected' : ''}>attention projections</option><option value="all"${targets === 'all' ? ' selected' : ''}>attention and FFN${isMoE(m) ? ' (every expert)' : ''}</option></select></label></p>
      ${figure(s, W, options.length * rowH + 8, 'fine-tuning memory')}
      <table class="data"><thead><tr><th>method</th><th class="r">trainable parameters</th><th class="r">weights + training state</th><th>made of</th></tr></thead><tbody>
      ${options.map(([name, bytes, note], i) => `<tr><td>${name}</td><td class="r">${fmtParams(i ? loraP : P.total)}${i ? ` (${fmtPct(loraP, P.total)})` : ''}</td><td class="r">${fmtBytes(bytes)}</td><td class="f">${note}</td></tr>`).join('')}</tbody></table>
      <p class="caption">${esc(m.name)}, rank ${rank}. Dashed line: one GPU of ${fmtBytes(ctx.mem)}. Activations come on top in every case: LoRA still runs the backward pass through all layers to reach the adapters, but stores no gradient or optimizer state for the frozen weights. After training, B·A can be added into W, so inference runs at the original speed.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    return `
      <p class="lead">Post-training updates a model that already exists. Full fine-tuning trains every parameter and needs the whole training state: 18 bytes per parameter in standard mixed precision. Low-rank adaptation (LoRA) keeps the weights frozen and adds a trainable update of low rank to selected matrices: W + B·A, with B and A thin. Only B and A get gradients and optimizer state, so the training state shrinks by orders of magnitude; QLoRA also stores the frozen weights in 4 bits.</p>
      <h2>A low-rank update</h2>${figMatrix(m, W)}
      <h2>Memory for each method</h2>${figMemory(m, ctx, W)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>LoRA: h = W<sub>0</sub>x + (α/r)·B·A·x, with W<sub>0</sub> ∈ ℝ<sup>d<sub>out</sub> × d<sub>in</sub></sup>, A ∈ ℝ<sup>r × d<sub>in</sub></sup> initialized from a Gaussian and B ∈ ℝ<sup>d<sub>out</sub> × r</sup> initialized to zero. B·A = 0 at the start, so training starts from the original model.</li>
        <li>Trainable parameters = Σ over adapted matrices r·(d<sub>in</sub> + d<sub>out</sub>).</li>
        <li>QLoRA: frozen linear-layer weights in 4-bit NormalFloat (NF4) with one absmax scale per 64 weights; the scales are stored as 8-bit floats in blocks of 256 with one FP32 scale each (double quantization): 4 + 8/64 + 32/(64·256) = 4.127 bits per weight. Weights are dequantized to BF16 for each multiplication. Hugging Face's bitsandbytes integration replaces only linear layers and keeps the output layer unquantized.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer">LoRA (Hu et al., 2021)</a> · <a href="https://arxiv.org/abs/2305.14314" target="_blank" rel="noreferrer">QLoRA (Dettmers et al., 2023)</a> · <a href="https://github.com/mlcommons/training/tree/master/llama2_70b_lora" target="_blank" rel="noreferrer">MLPerf Training Llama 2 70B LoRA reference implementation</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#lo-r').addEventListener('change', (e) => { rank = Number(e.target.value); rerender(); });
    root.querySelector('#lo-t').addEventListener('change', (e) => { targets = e.target.value; rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).lora = { render, mount };
})();
