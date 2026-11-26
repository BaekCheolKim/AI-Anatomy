// Chapter: Training loop.
// Figures: (1) one training step: what each stage reads and writes; (2) bytes per parameter of the
// training state and how many GPUs it fills; (3) saved activations per sequence, with and without
// recomputation; (4) total training compute against published GPU hours.
(function () {
  let optimizer = 'adamw', seq = 8192;

  function figStep(W) {
    const steps = [
      ['1 Forward', [['w', 'reads weights'], ['a', 'saves activations']]],
      ['2 Loss', [['a', 'logits vs next tokens']]],
      ['3 Backward', [['w', 'reads weights'], ['a', 'reads activations'], ['g', 'writes gradients']]],
      ['4 Update', [['g', 'reads gradients, optimizer state'], ['w', 'writes weights']]],
    ];
    const narrow = W < 560, colW = narrow ? W / 2 : W / 4, rowH = narrow ? 92 : 100;
    let s = '';
    steps.forEach(([name, items], i) => {
      const x = (i % (narrow ? 2 : 4)) * colW, y = Math.floor(i / (narrow ? 2 : 4)) * rowH;
      s += `<text x="${x}" y="${y + 14}" class="name"><tspan font-weight="700">${name}</tspan></text>`;
      items.forEach(([k, t], j) => { s += `<rect x="${x}" y="${y + 26 + j * 18}" width="10" height="10" class="${k}"/><text x="${x + 16}" y="${y + 35 + j * 18}" class="val">${t}</text>`; });
      if ((i % (narrow ? 2 : 4)) < (narrow ? 1 : 3)) s += `<text x="${x + colW - 18}" y="${y + 14}" class="axis">→</text>`;
    });
    const h = (narrow ? 2 : 1) * rowH;
    s += `<text x="0" y="${h - 4}" class="axis">↺ repeat with the next batch</text>`;
    return `${figure(s, W, h + 2, 'one training step')}
      <p class="caption">Colors: green, weights; orange, activations; vermillion, gradients and optimizer state. The forward pass keeps activations that the backward pass needs; the backward pass costs about twice the forward FLOPs, because each weight multiplication y = x·W needs both ∂L/∂x = ∂L/∂y·Wᵀ and ∂L/∂W = xᵀ·∂L/∂y.</p>`;
  }

  function figState(m, ctx, W) {
    const P = params(m), b = trainBytesPerParam(m, optimizer), total = P.total * b.total;
    const parts = [['w', 'BF16 weights', b.weights], ['g', 'FP32 gradients', b.grads], ['g-mid', 'FP32 master weights', b.master],
      ['g-dark', `${b.states === 1 ? 'momentum' : 'moments m, v'} (${b.momentDtype.toUpperCase()})`, b.optimizer]];
    const k = (W - 2) / 18;
    let x = 0, s = '';
    parts.forEach(([cls, name, v]) => { s += `<rect x="${x}" y="0" width="${v * k}" height="22" class="${cls} sep"><title>${name}: ${v} B</title></rect><text x="${x + 3}" y="38" class="val">${v} B</text>`; x += v * k; });
    const minGPUs = Math.ceil(total / ctx.mem), per = total / ctx.gpus;
    const legend = parts.map(([cls, name]) => `<span class="nw"><span class="key ${cls}"></span>${name}</span>`).join(' ');
    return `<p class="controls">Optimizer <select id="tr-opt"><option value="adamw"${optimizer === 'adamw' ? ' selected' : ''}>AdamW (2 states)</option><option value="muon"${optimizer === 'muon' ? ' selected' : ''}>Muon (1 state)</option></select></p>
      ${figure(s, W, 44, 'bytes per parameter')}
      <p class="caption legend">${legend}</p>
      <p class="status">${b.total} bytes per parameter × ${fmtParams(P.total)} parameters = <b>${fmtBytes(total)}</b> of training state for ${esc(m.name)}, before activations. That fills at least ${fmtN(minGPUs)} GPU${minGPUs > 1 ? 's' : ''} of ${fmtBytes(ctx.mem)}. Sharded evenly over the ${fmtN(ctx.gpus)} GPU${ctx.gpus > 1 ? 's' : ''} set at the top (ZeRO-3 / FSDP), each holds ${fmtBytes(per)}${per > ctx.mem ? ` (${fmtBytes(per - ctx.mem)} more than GPU memory)` : ''}.</p>
      <p class="caption">Mixed precision as in Megatron-LM: the forward and backward passes use BF16 weights; gradients accumulate in FP32; the optimizer updates an FP32 master copy and writes it back to BF16.${m.trainPrecision ? ` ${esc(m.trainPrecision.note)}` : ''} Muon keeps one momentum buffer per parameter instead of AdamW's two moments (used for Kimi K2).</p>`;
  }

  function figActivations(m, ctx, W) {
    const lens = [2048, 8192, 32768].filter((t) => t <= Math.max(m.context, 2048));
    const none = savedValuesPerToken(m, 'none') * 2, full = savedValuesPerToken(m, 'full') * 2;
    const max = none * Math.max(...lens), lw = W < 560 ? 0 : 236, bw = W - lw - (W < 560 ? 0 : 90), rowH = W < 560 ? 40 : 26;
    let s = '', r = 0;
    lens.forEach((t) => {
      [['no recomputation', none], ['full recomputation', full]].forEach(([name, perTok], j) => {
        const y = r * rowH + (W < 560 ? 14 : 0) + (j === 0 ? 4 : 0), v = perTok * t;
        s += `<text x="0" y="${W < 560 ? y : y + 13}" class="${j ? 'axis' : 'name'}">${j ? name : `${fmtN(t)} tokens, ${name}`}</text>`
          + `<rect x="${lw}" y="${W < 560 ? y + 4 : y + 2}" width="${Math.max(1, (v / max) * bw)}" height="14" class="a"/>`
          + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 13}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${fmtBytes(v)}</text>`;
        r++;
      });
    });
    return `${figure(s, W, r * rowH + (W < 560 ? 14 : 6), 'saved activations')}
      <p class="caption">Activations saved for the backward pass for one sequence on one GPU (BF16, approximate; the exact set depends on the kernels): ${fmtBytes(none)} per token without recomputation, ${fmtBytes(full)} with full recomputation, which keeps only each layer's input and runs the forward pass of a layer again during backward (one more forward pass of FLOPs). Splitting the model with tensor and sequence parallelism divides these per GPU.</p>`;
  }

  function figCompute(m, W) {
    const rows = window.MODELS.filter((x) => x.training?.tokens).map((x) => {
      const P = params(x), tr = x.training, f6 = 6 * P.matmul * tr.tokens;
      const ach = tr.gpuHours ? f6 / (tr.gpuHours * 3600) : null;
      return `<tr class="${x.id === m.id ? 'on' : ''}"><td>${esc(x.name)}</td><td class="r">${fmtSig(tr.tokens / 1e12)}T</td><td class="r">${fmtExp(f6)}</td>
        <td>${tr.flopsText ? esc(tr.flopsText) : '—'}</td><td>${tr.gpuHours ? `${fmtN(tr.gpuHours)} ${esc(tr.gpu)} GPU h` : '—'}</td><td class="r">${ach ? `${fmtSig(ach / 1e12)} TFLOP/s` : '—'}</td></tr>`;
    }).join('');
    return `<table class="data"><thead><tr><th>model</th><th class="r">tokens</th><th class="r">6·N·D FLOPs</th><th>published compute</th><th>published GPU hours</th><th class="r">FLOP/s per GPU</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="caption">6·N·D: 6 FLOPs per multiplied weight per token (forward 2, backward 4) × weights used per token (N; active weights for MoE) × training tokens (D). The last column divides it by the published GPU time: the sustained rate per GPU over the whole run, including restarts and evaluation. Compare it with a GPU's peak BF16 or FP8 rate to get the utilization. GPU hours are measured values from the authors; the model and token counts are computed from configs.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, P = params(m), tr = m.training;
    const tf = trainingFlops(m, 1, Math.min(seq, m.context));
    return `
      <p class="lead">Training repeats one step. A batch of token sequences runs forward through the network; the loss compares the predictions with the tokens that follow; the backward pass computes the gradient of the loss for every parameter; the optimizer updates every parameter from its gradient. Pre-training repeats this step until the training set has been read, usually once.</p>
      <p><b>${esc(m.name)}</b>: ${fmtParams(P.matmul)} weights are multiplied per token, so one training token costs about 6 × ${fmtParams(P.matmul)} = ${fmtFlops(6 * P.matmul)}, plus attention: ${fmtFlops(tf.perToken)} per token at sequence length ${fmtN(Math.min(seq, m.context))}.${tr?.tokens ? ` Over its ${esc(tr.text)}: ${fmtFlops(trainingFlops(m, tr.tokens, Math.min(seq, m.context)).total)}.` : ''}</p>
      <h2>One step</h2>${figStep(W)}
      <h2>Training state per parameter</h2>${figState(m, ctx, W)}
      <h2>Saved activations</h2>${figActivations(m, ctx, W)}
      <h2>Compute of whole training runs</h2>${figCompute(m, W)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>AdamW, per parameter w with gradient g: m ← β₁m + (1 − β₁)g; v ← β₂v + (1 − β₂)g²; w ← w − η(m̂ / (√v̂ + ε) + λw), with m̂, v̂ the bias-corrected moments.</li>
        <li>Training FLOPs ≈ 6·N·D + attention; with full recomputation 8·N·D.</li>
        <li>ZeRO stages shard over the data-parallel GPUs: 1, optimizer states; 2, also gradients; 3 (FSDP), also weights, which are then gathered layer by layer when used.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">ZeRO</a> · <a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noreferrer">AdamW</a> · <a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noreferrer">Llama 3 paper</a> · <a href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noreferrer">DeepSeek-V3 report</a>${tr?.gpuHoursUrl ? ` · <a href="${esc(tr.gpuHoursUrl)}" target="_blank" rel="noreferrer">${esc(m.name)} GPU hours</a>` : ''}.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#tr-opt').addEventListener('change', (e) => { optimizer = e.target.value; rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).training = { render, mount };
})();
