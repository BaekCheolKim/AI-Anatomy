// Chapter: Images as tokens.
// Figures: (1) an image of chosen size divided into patches and visual tokens; (2) the path from
// pixels to tokens with shapes, parameters and FLOPs; (3) tokens and cache bytes for common sizes.
(function () {
  let imgW = 1024, imgH = 768;

  // Qwen-VL processors resize both sides to multiples of patch × merge (32 px here).
  function plan(v, w, h) {
    const unit = v.patch * v.merge, W = Math.max(unit, Math.round(w / unit) * unit), H = Math.max(unit, Math.round(h / unit) * unit);
    const patches = (W / v.patch) * (H / v.patch), tokens = patches / (v.merge * v.merge);
    return { W, H, unit, patches, tokens, cols: W / unit, rows: H / unit };
  }

  function figGrid(m, v, W) {
    const p = plan(v, imgW, imgH), scale = Math.min((W - 2) / p.W, 360 / p.H), w = p.W * scale, h = p.H * scale;
    const fine = v.patch * scale, coarse = p.unit * scale;
    let s = `<rect x="0" y="0" width="${w}" height="${h}" class="img"/>`;
    if (fine >= 3) for (let x = fine; x < w - 0.5; x += fine) s += `<line x1="${x}" x2="${x}" y1="0" y2="${h}" class="patchline"/>`;
    if (fine >= 3) for (let y = fine; y < h - 0.5; y += fine) s += `<line x1="0" x2="${w}" y1="${y}" y2="${y}" class="patchline"/>`;
    for (let x = coarse; x < w - 0.5; x += coarse) s += `<line x1="${x}" x2="${x}" y1="0" y2="${h}" class="tokenline"/>`;
    for (let y = coarse; y < h - 0.5; y += coarse) s += `<line x1="0" x2="${w}" y1="${y}" y2="${y}" class="tokenline"/>`;
    s += `<rect x="0" y="0" width="${coarse}" height="${coarse}" class="onetoken"><title>one visual token: ${p.unit} × ${p.unit} pixels, ${v.merge * v.merge} patches</title></rect>`;
    return `<p class="controls">Image <input id="img-w" type="number" min="32" step="32" value="${imgW}" class="num-in"> × <input id="img-h" type="number" min="32" step="32" value="${imgH}" class="num-in"> pixels</p>
      ${figure(s, w, h, 'patches and visual tokens')}
      <p class="status">Resized to ${p.W} × ${p.H}: ${fmtN(p.patches)} patches of ${v.patch} × ${v.patch} pixels → ${fmtN(p.tokens)} visual tokens (${p.cols} × ${p.rows}), each covering ${p.unit} × ${p.unit} pixels. In the language model they take ${fmtBytes(p.tokens * kvBytesPerToken(m))} of KV cache per image (BF16).</p>
      <p class="caption">Thin lines: ${v.patch} × ${v.patch} patches, the vision encoder's input positions. Thick lines: groups of ${v.merge} × ${v.merge} patches, each merged into one token of the language model (outlined: one token).${fine < 3 ? ' Patch lines are omitted at this scale.' : ''}</p>`;
  }

  function figPath(m, v) {
    const p = plan(v, imgW, imgH), h = v.hidden, hm = h * v.merge * v.merge;
    const block = 2 * (2 * h) + (h * 3 * h + 3 * h) + (h * h + h) + (h * v.inter + v.inter) + (v.inter * h + h);
    const patchP = v.channels * v.temporalPatch * v.patch * v.patch * h + h;
    const merger = (post) => 2 * (post ? hm : h) + (hm * hm + hm) + (hm * v.outHidden + v.outHidden);
    const attnF = 4 * v.layers * p.patches * p.patches * h;
    const rows = [
      ['Patch embedding', `${v.channels} × ${v.temporalPatch} × ${v.patch} × ${v.patch} pixel values → ${fmtN(h)}`, `${fmtN(p.patches)} × ${fmtN(h)}`, patchP, 2 * patchP * p.patches, 'a convolution with stride = kernel; an image is repeated to fill the 2-frame time axis'],
      ['Position embedding', `${fmtN(v.posEmbed)} learned positions (${Math.sqrt(v.posEmbed)} × ${Math.sqrt(v.posEmbed)}), bilinearly interpolated to the patch grid and added to the patch embeddings`, `${fmtN(p.patches)} × ${fmtN(h)}`, v.posEmbed * h, 0, 'each encoder block also applies 2-D rotary embedding (row and column index) to its queries and keys'],
      [`${v.layers} encoder blocks`, `bidirectional attention over the image's patches + GELU MLP (${fmtN(h)} → ${fmtN(v.inter)} → ${fmtN(h)})`, `${fmtN(p.patches)} × ${fmtN(h)}`, v.layers * block, 2 * v.layers * block * p.patches + attnF, `attention part ≈ ${fmtFlops(attnF)} if every block attends over all ${fmtN(p.patches)} patches`],
      ['Merger', `concatenate ${v.merge} × ${v.merge} neighbours (${fmtN(hm)}), MLP → ${fmtN(v.outHidden)}`, `${fmtN(p.tokens)} × ${fmtN(v.outHidden)}`, merger(false), 2 * (merger(false) - 2 * h) * p.tokens, 'output width = the language model\'s H'],
    ];
    if (v.deepstack) rows.push(['DeepStack mergers', `${v.deepstack} more mergers on intermediate encoder layers`, `${v.deepstack} × ${fmtN(p.tokens)} × ${fmtN(v.outHidden)}`, v.deepstack * merger(true), 2 * v.deepstack * (merger(true) - 2 * hm) * p.tokens, 'added to the hidden states of the first language-model layers']);
    const tot = rows.reduce((s, r) => s + r[3], 0), fl = rows.reduce((s, r) => s + r[4], 0);
    return `<table class="data ops"><thead><tr><th>step</th><th>operation</th><th>output</th><th class="r">parameters</th><th class="r">FLOPs per image</th></tr></thead><tbody>
      ${rows.map(([n, f, out, prm, flops, note]) => `<tr><td><b>${n}</b></td><td class="f">${f}${note ? `<br><span class="note-inline">${note}</span>` : ''}</td><td class="mono shape">${out}</td><td class="r">${fmtParams(prm)}</td><td class="r">${flops ? fmtFlops(flops) : '—'}</td></tr>`).join('')}
      <tr><td><b>Total</b></td><td></td><td></td><td class="r"><b>${fmtParams(tot)}</b></td><td class="r"><b>${fmtFlops(fl)}</b></td></tr></tbody></table>
      <p class="caption">Vision encoder of ${esc(m.name)} for one ${p.W} × ${p.H} image. Its parameters (${fmtParams(params(m).vision)}) are ${fmtPct(params(m).vision, params(m).total)} of the model. After the merger, the ${fmtN(p.tokens)} visual tokens pass through all ${m.layers} language-model layers like text tokens: about ${fmtFlops(2 * params(m).matmul * p.tokens)} more.</p>`;
  }

  function figSizes(m, v, W) {
    const sizes = [[448, 448], [1024, 768], [1920, 1080], [3840, 2160]];
    return `<table class="data"><thead><tr><th>image</th><th class="r">patches</th><th class="r">visual tokens</th><th class="r">KV cache (BF16)</th></tr></thead><tbody>
      ${sizes.map(([w, h]) => { const p = plan(v, w, h); return `<tr><td>${w} × ${h}</td><td class="r">${fmtN(p.patches)}</td><td class="r">${fmtN(p.tokens)}</td><td class="r">${fmtBytes(p.tokens * kvBytesPerToken(m))}</td></tr>`; }).join('')}</tbody></table>
      <p class="caption">Tokens grow with the pixel count: a 3840 × 2160 frame becomes 8,160 visual tokens. Processors cap the pixel count per image (configurable), resizing larger images down before they are cut into patches.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, v = m.vision;
    const lead = '<p class="lead">A vision-language model turns an image into tokens of the same width as text tokens. The image is cut into square patches; a vision transformer (ViT) encodes the patches; groups of neighbouring patches are merged; a small network maps each group to one vector of the language model\'s width. These vectors are placed in the token sequence between the text tokens, and the language model processes them like text.</p>';
    if (!v) {
      const vl = window.MODELS.filter((x) => x.vision).map((x) => `<a href="#" data-model="${x.id}">${esc(x.name)}</a>`).join(' or ');
      return `${lead}<p><b>${esc(m.name)}</b> reads text only. Select ${vl} to see their image path.</p>`;
    }
    return `${lead}
      <p><b>${esc(m.name)}</b>: a ${v.layers}-layer vision encoder (width ${fmtN(v.hidden)}) on ${v.patch} × ${v.patch}-pixel patches, ${v.merge} × ${v.merge} patches merged per token, so one visual token covers ${v.patch * v.merge} × ${v.patch * v.merge} pixels.${v.deepstack ? ` It also merges features of ${v.deepstack} intermediate encoder layers and adds them to the language model's first layers (DeepStack).` : ''}${m.mrope ? ' Visual tokens get time, height and width positions, encoded in interleaved dimension pairs of the rotary embedding (interleaved M-RoPE).' : ''} Video frames enter in pairs (temporal patch ${v.temporalPatch}).</p>
      <h2>One image, cut into tokens</h2>${figGrid(m, v, W)}
      <h2>From pixels to tokens</h2>${figPath(m, v)}
      <h2>Tokens per image size</h2>${figSizes(m, v, W)}
      <p class="sources">Sources: ${configLink(m, 'config.json (vision_config)')} · <a href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noreferrer">ViT (Dosovitskiy et al.)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    ['w', 'h'].forEach((k) => {
      const el = root.querySelector(`#img-${k}`);
      if (el) el.addEventListener('change', () => { const v = Math.min(8192, Math.max(32, Number(el.value) || 32)); if (k === 'w') imgW = v; else imgH = v; rerender(); });
    });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).images = { render, mount };
})();
