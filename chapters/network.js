// Chapter: Inside the network.
// Figures: (1) the layer pattern of every model (token-mixing type over FFN type, one column per
// layer), (2) one token through one layer of the selected model: each operation with its formula,
// output shape, parameters and FLOPs, (3) where a token's FLOPs go across the whole network.
(function () {
  let layerSel = {};   // per model id: selected layer index

  const MIXER_NAME = { full: 'full attention', sliding: 'sliding-window attention', linear: 'Gated DeltaNet (linear attention)' };
  const ffnKind = (m, i) => (isMoE(m) && i >= m.ffn.denseLayers ? 'moe' : 'dense');
  const mixerName = (m, t) => (t !== 'linear' && m.attention.kind === 'MLA' ? 'multi-head latent attention' : t === 'full' && m.attention.outputGate ? 'gated full attention' : MIXER_NAME[t]);

  // Operations of layer i for one token. params: weights used for this token; flops: 2 per weight.
  function layerOps(m, i) {
    const H = m.hidden, a = m.attention, t = layerTypes(m)[i], ops = [];
    const n = fmtN, normF = m.norm === 'RMSNorm' ? (m.normGainPlusOne ? 'x̂ = x / √(mean(x²) + ε) ⊙ (1 + w), the gain stored as w = g − 1' : 'x̂ = x / √(mean(x²) + ε) ⊙ g') : 'x̂ = (x − μ) / √(σ² + ε) ⊙ g + b';
    const yarn = m.ropeScaling?.type === 'yarn' ? 1 + 0.1 * Math.log(m.ropeScaling.factor) : null;   // YaRN attention factor
    const norm = (label) => ({ name: `${m.norm}, ${label}`, f: normF, out: `${n(H)}`, p: normParams(m) });
    ops.push(norm('before token mixing'));
    if (t === 'linear') {
      const g = m.linearAttention, kT = g.keyHeads * g.keyDim, vT = g.valueHeads * g.valueDim;
      ops.push({ name: 'Input projection', f: 'q, k, v, z, b, a = x̂·W<sub>in</sub>', out: `q, k ${g.keyHeads} × ${g.keyDim}; v, z ${g.valueHeads} × ${g.valueDim}; b, a ${g.valueHeads}`, p: H * (2 * kT + 2 * vT) + H * 2 * g.valueHeads });
      ops.push({ name: `Causal convolution, width ${g.conv}`, f: 'q, k, v ← SiLU(depthwise conv over the last 4 positions)', out: 'same', p: (2 * kT + vT) * g.conv, note: `keeps the last ${g.conv - 1} inputs per sequence` });
      ops.push({ name: 'Gated delta rule', f: `q, k L2-normalized per head, q × 1/√${g.keyDim}; α = exp(−exp(A_log)·softplus(a + dt_bias)), β = σ(b); per value head: S ← α·S·(I − β k kᵀ) + β v kᵀ; o = S q`, out: `${g.valueHeads} × ${g.valueDim}`, p: 2 * g.valueHeads,
        note: `state S: ${g.valueHeads} × ${g.keyDim} × ${g.valueDim} values per sequence, the same size at every length; each query/key head serves ${g.valueHeads / g.keyHeads} consecutive value heads`, mix: true });
      ops.push({ name: 'Gated RMSNorm', f: 'o ← RMSNorm(o) ⊙ SiLU(z)', out: `${g.valueHeads} × ${g.valueDim}`, p: g.valueDim });
      ops.push({ name: 'Output projection', f: 'o·W<sub>out</sub>', out: n(H), p: vT * H });
    } else if (a.kind === 'MLA') {
      const dqk = a.nopeDim + a.ropeDim;
      ops.push({ name: 'Query compression', f: 'c<sub>Q</sub> = RMSNorm(x̂·W<sub>DQ</sub>)', out: n(a.qRank), p: H * a.qRank + a.qRank });
      ops.push({ name: 'Query up-projection', f: 'q = c<sub>Q</sub>·W<sub>UQ</sub>, RoPE on the last 64 dims', out: `${a.heads} × ${dqk}`, p: a.qRank * a.heads * dqk });
      ops.push({ name: 'Key/value compression', f: 'c<sub>KV</sub> = RMSNorm(x̂·W<sub>DKV</sub>); k<sub>R</sub> = RoPE(x̂·W<sub>KR</sub>)', out: `${a.kvRank} + ${a.ropeDim}`, p: H * (a.kvRank + a.ropeDim) + a.kvRank,
        note: `cached per position: c_KV and k_R, ${a.kvRank + a.ropeDim} values` });
      ops.push({ name: 'Key/value up-projection', f: 'k<sub>C</sub> = c<sub>KV</sub>·W<sub>UK</sub>, v = c<sub>KV</sub>·W<sub>UV</sub>; k = [k<sub>C</sub>, k<sub>R</sub>]', out: `${a.heads} × (${a.nopeDim} + ${a.vDim})`, p: a.kvRank * a.heads * (a.nopeDim + a.vDim),
        note: 'in decode, W_UK and W_UV are multiplied into the query and output side instead (absorbed)' });
      ops.push({ name: 'Attention', f: `A = softmax(${yarn ? 'm²·' : ''}q·kᵀ / √${dqk}) over positions ≤ t; o = A·v${yarn ? `; m = 1 + 0.1·ln ${m.ropeScaling.factor} ≈ ${yarn.toFixed(3)} (YaRN)` : ''}`, out: `${a.heads} × ${a.vDim}`, p: 0, mix: true });
      ops.push({ name: 'Output projection', f: 'o·W<sub>O</sub>', out: n(H), p: a.heads * a.vDim * H });
    } else {
      const q = a.heads * a.headDim, k = a.kvHeads * a.headDim, qOut = q * (a.outputGate ? 2 : 1);
      const bias = a.bias === 'all' || a.bias === 'qkv' ? qOut + 2 * k : 0;
      const grp = a.heads / a.kvHeads;
      ops.push({ name: a.kind === 'MHA' ? 'Q, K, V projections' : 'Q, K, V projections (grouped)', f: `q = x̂·W<sub>Q</sub>, k = x̂·W<sub>K</sub>, v = x̂·W<sub>V</sub>${bias ? ' (+ biases)' : ''}${a.outputGate ? '; W<sub>Q</sub> also outputs a gate g' : ''}`,
        out: `q ${a.heads} × ${a.headDim}; k, v ${a.kvHeads} × ${a.headDim}${a.outputGate ? `; g ${a.heads} × ${a.headDim}` : ''}`, p: H * qOut + 2 * H * k + bias,
        note: a.kind === 'GQA' ? `each K/V head is shared by ${grp} query heads` : 'one K/V head per query head' });
      if (a.qkNorm) ops.push({ name: 'QK-norm', f: 'RMSNorm on each query and key head', out: 'same', p: 2 * a.headDim });
      if (m.position === 'rope') ops.push({ name: 'Rotary position embedding', f: `rotate pairs of dims of q and k by angle position × θ<sub>i</sub>${m.ropeFraction ? ` (${m.ropeFraction * 100}% of each head's dims)` : ''}${yarn ? `; q and k also × ${yarn.toFixed(3)} (YaRN: 1 + 0.1·ln ${m.ropeScaling.factor}), so q·k × ${(yarn * yarn).toFixed(3)}` : ''}`, out: 'same', p: 0 });
      const win = t === 'sliding' ? `the last ${a.window} positions` : 'positions ≤ t';
      ops.push({ name: t === 'sliding' ? `Attention, window ${a.window}` : 'Attention', f: `A = softmax(q·kᵀ / √${a.headDim}${a.sinks ? ', with one sink logit per head' : ''}) over ${win}; o = A·v`, out: `${a.heads} × ${a.headDim}`, p: a.sinks ? a.heads : 0, mix: true,
        note: `cached per position: k and v, ${2 * a.kvHeads * a.headDim} values${t === 'sliding' ? `, at most ${a.window} positions` : ''}` });
      if (a.outputGate) ops.push({ name: 'Output gate', f: 'o ← o ⊙ σ(g)', out: 'same', p: 0 });
      ops.push({ name: 'Output projection', f: `o·W<sub>O</sub>${a.bias === 'all' ? ' + b' : ''}`, out: n(H), p: q * H + (a.bias === 'all' ? H : 0) });
    }
    ops.push({ name: 'Residual add', f: 'h ← h + mixer(norm(h))', out: n(H), p: 0, add: true });
    ops.push(norm('before the FFN'));
    const f = m.ffn;
    if (ffnKind(m, i) === 'dense') {
      const F = isMoE(m) ? f.denseInter : f.inter;
      if (f.gated) {
        ops.push({ name: 'Gate and up projections', f: 'u = x̂·W<sub>gate</sub>, v = x̂·W<sub>up</sub>', out: `2 × ${n(F)}`, p: 2 * H * F });
        ops.push({ name: 'SwiGLU', f: 'SiLU(u) ⊙ v', out: n(F), p: 0 });
        ops.push({ name: 'Down projection', f: '·W<sub>down</sub>', out: n(H), p: F * H });
      } else {
        ops.push({ name: 'Up projection', f: `x̂·W<sub>1</sub>${f.bias ? ' + b<sub>1</sub>' : ''}, then GELU`, out: n(F), p: H * F + (f.bias ? F : 0) });
        ops.push({ name: 'Down projection', f: `·W<sub>2</sub>${f.bias ? ' + b<sub>2</sub>' : ''}`, out: n(H), p: F * H + (f.bias ? H : 0) });
      }
    } else {
      const ml = moeLayer(m);
      ops.push({ name: 'Router', f: f.router === 'sigmoid-grouped' ? `s = σ(x̂·W<sub>r</sub>); keep the ${f.topkGroups} of ${f.groups} groups, then the ${f.topK} experts, ranked by s + b (b only for selection); g<sub>e</sub> = ${f.routedScale}·s<sub>e</sub> / Σ<sub>selected</sub> s`
        : f.router === 'topk-softmax' ? `s = x̂·W<sub>r</sub> + b; keep the ${f.topK} largest; g = softmax over those ${f.topK}` : `p = softmax(x̂·W<sub>r</sub>); keep the ${f.topK} largest${f.normTopK ? ', rescaled to sum to 1' : ''}`, out: `${f.experts} scores → ${f.topK} experts`, p: ml.router });
      ops.push({ name: `${f.topK} routed experts`, f: f.swigluLimit ? `y<sub>e</sub> = ((a ⊙ σ(1.702·a)) ⊙ (u + 1))·W<sub>down,e</sub> + b; a = min(x̂·W<sub>gate,e</sub> + b, ${f.swigluLimit}), u = clamp(x̂·W<sub>up,e</sub> + b, −${f.swigluLimit}, ${f.swigluLimit})` : 'y<sub>e</sub> = (SiLU(x̂·W<sub>gate,e</sub>) ⊙ x̂·W<sub>up,e</sub>)·W<sub>down,e</sub>', out: `${f.topK} × ${n(H)}`, p: f.topK * ml.perExpert,
        note: `${f.experts - f.topK} other experts of this layer, ${fmtParams((f.experts - f.topK) * ml.perExpert)} parameters, are stored but not used for this token` });
      if (f.shared) ops.push({ name: 'Shared expert', f: `same FFN shape${f.sharedGate ? ', output × σ(x̂·w<sub>s</sub>)' : ''}`, out: n(H), p: ml.shared });
      ops.push({ name: 'Weighted sum', f: `y = Σ g<sub>e</sub>·y<sub>e</sub>${f.shared ? ' + shared' : ''}`, out: n(H), p: 0 });
    }
    ops.push({ name: 'Residual add', f: 'h ← h + FFN(norm(h))', out: n(H), p: 0, add: true });
    return ops;
  }

  // ---- figure 1: layer patterns of all models ----
  function figPatterns(m, W) {
    const narrow = W < 560, lw = narrow ? 0 : 168, maxL = Math.max(...window.MODELS.map((x) => x.layers));
    const step = (W - lw) / maxL, cw = Math.max(1.5, step - (step > 3 ? 1 : 0.5)), rowH = narrow ? 50 : 34;
    const sel = layerSel[m.id] ?? 0;
    let s = '';
    window.MODELS.forEach((mm, r) => {
      const y0 = r * rowH + (narrow ? 16 : 2), on = mm.id === m.id, types = layerTypes(mm);
      s += `<g class="row${on ? ' on' : ''}"${on ? '' : ` data-model="${mm.id}"`}><rect x="0" y="${y0 - (narrow ? 14 : 0)}" width="${W}" height="${rowH - 2}" class="hit"/>`
        + `<text x="0" y="${narrow ? y0 - 2 : y0 + 17}" class="name">${esc(mm.name)}</text>`;
      types.forEach((t, i) => {
        const x = lw + i * step, mix = t === 'linear' ? 'lt-lin' : t === 'sliding' ? 'lt-sw' : 'lt-full', ffn = ffnKind(mm, i) === 'moe' ? 'lt-moe' : 'lt-dense';
        const pick = on && i === sel ? ' picked' : '';
        s += `<g${on ? ` data-layer="${i}"` : ''}><title>${esc(mm.name)} layer ${i + 1}: ${mixerName(mm, t)} + ${ffnKind(mm, i) === 'moe' ? 'MoE' : 'dense'} FFN</title>`
          + `<rect x="${x}" y="${y0 + 2}" width="${cw}" height="11" class="${mix}${pick}"/><rect x="${x}" y="${y0 + 15}" width="${cw}" height="11" class="${ffn}${pick}"/></g>`;
      });
      s += '</g>';
    });
    const legend = `<span class="nw">upper: <span class="key lt-full"></span>full or latent attention</span> <span class="nw"><span class="key lt-sw"></span>sliding window</span> <span class="nw"><span class="key lt-lin"></span>linear attention</span> · <span class="nw">lower: <span class="key lt-dense"></span>dense FFN</span> <span class="nw"><span class="key lt-moe"></span>MoE</span>`;
    return `${figure(s, W, window.MODELS.length * rowH + (narrow ? 16 : 0), 'layer patterns')}
      <p class="caption legend">${legend}</p>
      <p class="caption">One column per layer: the upper cell is the token-mixing sub-layer, the lower cell the FFN. Row length is proportional to the number of layers. Click a layer of the selected model to follow a token through it; click another row to select that model.</p>`;
  }

  // ---- figure 2: one token through layer i ----
  function figOps(m, W) {
    const i = Math.min(layerSel[m.id] ?? 0, m.layers - 1), ops = layerOps(m, i), t = layerTypes(m)[i];
    const total = ops.reduce((s, o) => s + o.p, 0), maxP = Math.max(...ops.map((o) => o.p));
    const rows = ops.map((o, k) => `<tr class="${o.add ? 'addrow' : ''}"><td class="n">${k + 1}</td><td><b>${esc(o.name)}</b><br><span class="f">${o.f}</span>${o.note ? `<br><span class="note-inline">${esc(o.note)}</span>` : ''}</td>
      <td class="mono shape">${esc(o.out)}</td><td class="r">${o.p ? fmtParams(o.p) : '—'}${o.p ? `<span class="pbar-inline" style="width:${Math.max(2, (o.p / maxP) * 60)}px"></span>` : ''}</td></tr>`).join('');
    const kinds = [];
    layerTypes(m).forEach((tt, j) => { const key = `${tt}|${ffnKind(m, j)}`; if (!kinds.some((x) => x.key === key)) kinds.push({ key, j, label: `${mixerName(m, tt)} + ${ffnKind(m, j) === 'moe' ? 'MoE' : 'dense FFN'}` }); });
    const picker = kinds.length > 1 ? `<p class="controls">Layer kinds in ${esc(m.name)}: ${kinds.map((k) => `<a href="#" data-layer="${k.j}" class="${layerTypes(m)[i] === k.key.split('|')[0] && ffnKind(m, i) === k.key.split('|')[1] ? 'on' : ''}">${esc(k.label)} (layer ${k.j + 1})</a>`).join(' · ')}</p>` : '';
    return `${picker}<p class="status">Layer ${i + 1} of ${m.layers}: ${mixerName(m, t)} + ${ffnKind(m, i) === 'moe' ? 'MoE' : 'dense'} FFN. One token enters as h ∈ ℝ<sup>${fmtN(m.hidden)}</sup> and leaves with the same shape.</p>
      <table class="data ops"><thead><tr><th>#</th><th>operation</th><th>output for one token</th><th class="r">weights used</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="caption">Weights used by one token in this layer: ${fmtParams(total)}, so ${fmtFlops(2 * total)} for the weight multiplications (2 FLOPs per weight). Attention over earlier positions adds FLOPs that grow with the context; the linear-attention state update does not.</p>`;
  }

  // ---- figure 3: where one token's FLOPs go ----
  function figFlops(m, W) {
    const P = params(m), ctxs = [1024, 8192, 32768, 131072].filter((c) => c <= m.context);
    const head = 2 * (m.tied ? P.embed : P.head);
    const max = Math.max(...ctxs.map((c) => forwardFlopsPerToken(m, c, 'decode').total));
    const lw = W < 560 ? 0 : 110, bw = W - lw - (W < 560 ? 0 : 90), rowH = W < 560 ? 44 : 30;
    let s = '';
    ctxs.forEach((c, r) => {
      const f = forwardFlopsPerToken(m, c, 'decode'), y = r * rowH + (W < 560 ? 16 : 2), k = bw / max;
      const body = f.weights - head;
      s += `<text x="0" y="${W < 560 ? y : y + 14}" class="name">${fmtN(c)} positions</text>`
        + `<rect x="${lw}" y="${(W < 560 ? y + 4 : y + 2)}" width="${body * k}" height="16" class="seg0 sep"><title>weights in the layers: ${fmtFlops(body)}</title></rect>`
        + `<rect x="${lw + body * k}" y="${(W < 560 ? y + 4 : y + 2)}" width="${head * k}" height="16" class="seg1 sep"><title>output layer: ${fmtFlops(head)}</title></rect>`
        + `<rect x="${lw + f.weights * k}" y="${(W < 560 ? y + 4 : y + 2)}" width="${Math.max(0.5, f.attention * k)}" height="16" class="actbar sep"><title>attention over ${fmtN(c)} positions: ${fmtFlops(f.attention)}</title></rect>`
        + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 14}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${fmtFlops(f.total)}</text>`;
    });
    return `${figure(s, W, ctxs.length * rowH + (W < 560 ? 16 : 0), 'FLOPs per token')}
      <p class="caption">Forward FLOPs to generate one token after the given number of positions: dark, weights of the ${m.layers} layers; light, the output layer (${fmtFlops(head)}); orange, attention over the cached positions (scores and weighted sum).${m.attention.kind === 'MLA' ? ' Multi-head latent attention is counted in its absorbed form, which decode kernels use: 2·(576 + 512) FLOPs per head per position.' : ''}</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, P = params(m), types = layerTypes(m);
    const counts = {};
    types.forEach((t, i) => { const k = `${mixerName(m, t)} + ${ffnKind(m, i) === 'moe' ? 'MoE' : 'dense FFN'}`; counts[k] = (counts[k] || 0) + 1; });
    const summary = Object.entries(counts).map(([k, v]) => `${v} × ${k}`).join(', ');
    return `
      <p class="lead">A transformer is a stack of layers between the embedding table and the output layer. Each position is carried through the stack as a vector of H values, the residual stream. Every layer adds two updates to it: a token-mixing sub-layer, which reads other positions (attention), and a feed-forward sub-layer (FFN), which processes each position on its own. Each sub-layer reads a normalized copy of the stream and adds its output back.</p>
      <p><b>${esc(m.name)}</b>: H = ${fmtN(m.hidden)}, ${m.layers} layers: ${summary}. ${m.position === 'learned' ? 'Positions enter as learned vectors added to the token embeddings.' : `Positions enter through rotary embeddings (θ = ${fmtN(m.ropeTheta)}) applied to queries and keys.`} Parameters: ${fmtParams(P.total)}.</p>
      <h2>Layer patterns</h2>${figPatterns(m, W)}
      <h2>One token through one layer</h2>${figOps(m, W)}
      <h2>FLOPs for one generated token</h2>${figFlops(m, W)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Pre-norm residual layer: h ← h + Mix(Norm(h)); h ← h + FFN(Norm(h)). After the last layer: z = Norm(h)·W<sub>out</sub>.</li>
        <li>RMSNorm: x / √(mean(x²) + ε) ⊙ g (ε = ${m.normEps}). LayerNorm subtracts the mean and adds a bias as well.${m.normGainPlusOne ? ' Qwen3.5 stores w = g − 1 and computes x / √(mean(x²) + ε) ⊙ (1 + w).' : ''}</li>
        <li>Rotary embedding: d/2 pairs of dims of q and k are rotated, pair i by angle p·θ<sup>−2i/d</sup> at position p (θ = rope_theta). The RoPE paper and DeepSeek-V3 pair dims 2i and 2i+1; the gpt-oss and Hugging Face Qwen code pair dim i with dim i + d/2, the same function up to a fixed permutation of dims. A query at position m and a key at position n then interact only through m − n.</li>
        <li>Weights used per token in the whole network: ${fmtParams(P.matmul)} (the embedding lookup reads one row and multiplies nothing), so about ${fmtFlops(2 * P.matmul)} per token before attention.</li>
      </ul></details>
      <p class="sources">Sources: ${configLink(m, 'config.json', true)} · <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Attention Is All You Need</a> · <a href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noreferrer">RoPE</a> · <a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noreferrer">GQA</a> · <a href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noreferrer">MLA (DeepSeek-V2)</a> · <a href="https://arxiv.org/abs/2412.06464" target="_blank" rel="noreferrer">Gated DeltaNet</a> · <a href="https://arxiv.org/abs/2505.06708" target="_blank" rel="noreferrer">Gated Attention</a> · <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py" target="_blank" rel="noreferrer">Qwen3.5 modeling code</a> · <a href="https://arxiv.org/abs/2508.10925" target="_blank" rel="noreferrer">gpt-oss model card</a> · <a href="https://arxiv.org/abs/2002.05202" target="_blank" rel="noreferrer">SwiGLU</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelectorAll('[data-layer]').forEach((el) => el.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation();
      layerSel[ctx.m.id] = Number(el.dataset.layer);
      rerender();
    }));
  }

  (window.CHAPTERS = window.CHAPTERS || {}).network = { render, mount };
})();
