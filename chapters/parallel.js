// Chapter: Splitting across GPUs (training).
// Controls: the five parallel degrees, GPUs per node, ZeRO stage, recomputation, batch.
// Figures: (1) the GPUs of the first nodes with their coordinates in each dimension; (2) memory per
// GPU; (3) bytes each GPU sends per step for every collective, inside and between nodes, with
// optional times when the reader enters link bandwidths.
(function () {
  const layouts = {};                 // per model id: the reader's current layout
  let bwIntra = '', bwInter = '';     // optional link bandwidths in GB/s per GPU, per direction

  const DIM = {
    dp: 'Data parallel (DP)', tp: 'Tensor parallel (TP)', pp: 'Pipeline parallel (PP)', ep: 'Expert parallel (EP)', cp: 'Context parallel (CP)',
    dpcp: 'Data parallel (DP × CP)', edp: 'Data parallel, expert copies',
  };
  const OPTS = {
    dp: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], tp: [1, 2, 4, 8, 16], pp: [1, 2, 4, 8, 16, 32], ep: [1, 2, 4, 8, 16, 32, 64, 128, 256],
    cp: [1, 2, 4, 8, 16], gpusPerNode: [4, 8], zero: [0, 1, 2, 3], recompute: ['none', 'full'], seq: [1024, 2048, 4096, 8192, 16384, 32768, 131072],
    mbs: [1, 2, 4, 8, 16], optimizer: ['adamw', 'muon'],
  };
  const LABEL = { dp: 'DP', tp: 'TP', pp: 'PP', ep: 'EP', cp: 'CP', gpusPerNode: 'GPUs per node', zero: 'ZeRO', recompute: 'recompute', seq: 'sequence', mbs: 'micro-batch', optimizer: 'optimizer' };

  const layoutOf = (m) => (layouts[m.id] ||= { optimizer: 'adamw', ...m.layouts.training });

  function controls(m, S) {
    const sel = (k) => {
      const list = OPTS[k].includes(S[k]) ? OPTS[k] : [...OPTS[k], S[k]].sort((a, b) => a - b);
      return `<label class="nw">${LABEL[k]} <select data-par="${k}">${list.map((o) => `<option value="${o}"${o === S[k] ? ' selected' : ''}>${o}</option>`).join('')}</select></label>`;
    };
    const keys = ['dp', 'tp', 'pp', ...(isMoE(m) ? ['ep'] : []), 'cp', 'gpusPerNode', 'zero', 'recompute', 'seq', 'mbs'];
    return `<p class="controls parctl">${keys.map(sel).join(' ')} <label class="nw">global batch <input type="number" min="1" data-par="gbs" value="${S.gbs}" class="num-in"> sequences</label> <a href="#" id="par-reset">reset</a></p>`;
  }

  function figNodes(m, S, W) {
    const per = S.gpusPerNode, N = S.dp * S.tp * S.pp * S.cp, show = Math.min(N, per * 2), nodes = Math.ceil(show / per);
    const cols = per, cellW = Math.min(150, Math.floor((W - 10) / cols)), cellH = isMoE(m) ? 70 : 58, gap = 18;
    let s = '';
    for (let r = 0; r < show; r++) {
      const node = Math.floor(r / per), i = r % per, x = i * cellW, y = node * (cellH + gap) + 16;
      const tp = r % S.tp, cp = Math.floor(r / S.tp) % S.cp, dp = Math.floor(r / (S.tp * S.cp)) % S.dp, pp = Math.floor(r / (S.tp * S.cp * S.dp));
      const ep = S.ep <= S.dp * S.cp ? Math.floor(r / S.tp) % S.ep : r % S.ep;
      if (i === 0) s += `<text x="0" y="${y - 5}" class="axis">node ${node + 1}</text>`;
      s += `<rect x="${x + 1}" y="${y}" width="${cellW - 4}" height="${cellH}" class="gpu"/><text x="${x + 6}" y="${y + 15}" class="name">GPU ${r}</text>`
        + `<text x="${x + 6}" y="${y + 30}" class="val">TP ${tp} · CP ${cp}</text><text x="${x + 6}" y="${y + 44}" class="val">DP ${dp} · PP ${pp}</text>`
        + (isMoE(m) ? `<text x="${x + 6}" y="${y + 58}" class="val">EP ${ep}</text>` : '');
    }
    return `${figure(s, W, nodes * (cellH + gap) + 4, 'GPUs and their coordinates')}
      <p class="caption">The first ${show} of ${fmtN(N)} GPUs with their index in each dimension. Ranks are numbered with TP fastest, then CP, DP and PP, so groups of consecutive ranks share a node: here a TP group ${S.tp <= per ? 'fits inside one node' : 'spans nodes'}${S.pp > 1 ? `, and consecutive pipeline stages are ${S.tp * S.cp * S.dp} ranks apart` : ''}.</p>`;
  }

  function figMemory(m, S, ctx, W) {
    const rows = trainingMemoryPerGPU(m, S), total = rows.reduce((s, r) => s + r.bytes, 0), mem = ctx.mem;
    const k = (W - 4) / Math.max(total, mem) / 1.04;
    let x = 0, s = '';
    rows.forEach((r) => { s += `<rect x="${x}" y="20" width="${Math.max(0.5, r.bytes * k)}" height="24" class="${r.kind} sep"><title>${r.name}: ${fmtBytes(r.bytes)}</title></rect>`; x += r.bytes * k; });
    s += `<line x1="${mem * k}" x2="${mem * k}" y1="10" y2="54" class="memline"/><text x="${mem * k}" y="8" text-anchor="${mem * k > W - 80 ? 'end' : 'middle'}" class="axis">GPU memory ${fmtBytes(mem)}</text>`;
    s += `<text x="0" y="68" class="val">${fmtBytes(total)} per GPU · ${total <= mem ? `${fmtBytes(mem - total)} left` : `${fmtBytes(total - mem)} over`}</text>`;
    return `${figure(s, W, 74, 'memory per GPU')}
      <table class="data"><thead><tr><th>tensor</th><th class="r">per GPU</th><th>split</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td><span class="key ${r.kind}"></span>${r.name}${r.approx ? ' ≈' : ''}</td><td class="r">${fmtBytes(r.bytes)}</td><td class="f">${esc(r.note)}</td></tr>`).join('')}</tbody></table>
      <p class="caption">Per GPU on the first pipeline stage, which keeps the most micro-batches of activations. ≈: estimates; framework workspaces and memory fragmentation are not included.</p>`;
  }

  function figCollectives(m, S) {
    const rows = trainingCollectiveRows(m, S), bi = Number(bwIntra) * 1e9, be = Number(bwInter) * 1e9, times = bi > 0 || be > 0;
    const t = (r) => (r.intra ? (bi > 0 ? r.intra / bi : NaN) : 0) + (r.inter ? (be > 0 ? r.inter / be : NaN) : 0);
    const ft = (v) => (Number.isNaN(v) ? '—' : v >= 1 ? `${v.toFixed(2)} s` : `${(v * 1e3).toFixed(v >= 0.1 ? 0 : 1)} ms`);
    const body = rows.map((r) => `<tr><td><b>${DIM[r.dim]}</b><br><span class="f">${esc(r.op)}: ${esc(r.when)}</span></td>
      <td class="r">${r.n} GPUs<br><span class="f">${r.r}/node, ${r.nodes} node${r.nodes > 1 ? 's' : ''}</span></td><td class="r">${fmtN(r.count)} × ${fmtBytes(r.M)}</td>
      <td class="r">${fmtBytes(r.intra)}</td><td class="r ${r.inter ? 'inter' : ''}">${fmtBytes(r.inter)}</td>${times ? `<td class="r">${ft(t(r))}</td>` : ''}</tr>`).join('');
    const sum = (k) => rows.reduce((s, r) => s + r[k], 0);
    return `<p class="controls">Optional, for times: inside a node <input id="bw-intra" type="number" min="0" step="any" value="${bwIntra}" class="num-in" placeholder="—"> GB/s, between nodes <input id="bw-inter" type="number" min="0" step="any" value="${bwInter}" class="num-in" placeholder="—"> GB/s per GPU, each direction</p>
      <table class="data"><thead><tr><th>collective</th><th class="r">group</th><th class="r">count × message</th><th class="r">sent inside node</th><th class="r">sent between nodes</th>${times ? '<th class="r">time</th>' : ''}</tr></thead>
      <tbody>${body}<tr><td><b>Total per GPU per step</b></td><td></td><td></td><td class="r"><b>${fmtBytes(sum('intra'))}</b></td><td class="r"><b>${fmtBytes(sum('inter'))}</b></td>${times ? `<td class="r"><b>${ft(rows.reduce((s, r) => s + t(r), 0))}</b></td>` : ''}</tr></tbody></table>
      <p class="caption">Bytes each GPU sends per training step. A group of n GPUs with r on each node first exchanges inside the node, then between nodes among GPUs with the same position in their node: all-gather and reduce-scatter send (r−1)/r·M inside and (N−1)/N·M/r between N nodes, all-reduce twice that; all-to-all sends (n−r)/n·M between nodes. Times, when bandwidths are entered, divide bytes by bandwidth with no overlap with computation: a lower bound. ${isMoE(m) && S.ep > 1 ? 'The expert all-to-all assumes every routed copy is sent separately in BF16; node-limited routing and FP8 dispatch (DeepSeek-V3) send less between nodes.' : ''}</p>`;
  }

  function dimList(m, S, L) {
    const bubble = S.pp > 1 ? ` With ${fmtN(L.micro)} micro-batches per step and 1F1B scheduling, the pipeline is idle (PP − 1)/(m + PP − 1) = ${fmtPct(S.pp - 1, L.micro + S.pp - 1)} of the step.` : '';
    return `<ul class="formulas">
      <li><b>DP</b> splits the batch: each of ${S.dp} copies processes ${fmtN(S.gbs / S.dp)} sequences per step, then gradients are summed across copies. ZeRO stage ${S.zero}: ${['every copy keeps full gradients and optimizer state', 'optimizer state split across copies', 'optimizer state and gradients split', 'optimizer state, gradients and weights split; weights gathered layer by layer (FSDP)'][S.zero]}.</li>
      <li><b>TP</b> splits each weight matrix over ${S.tp} GPU${S.tp > 1 ? 's' : ''}: attention heads and FFN columns per GPU; partial results are combined after the attention output and FFN down projections, 2 times per layer in the forward pass and 2 in the backward pass. Sequence parallelism also splits the norms and dropout along the sequence.</li>
      <li><b>PP</b> splits the ${m.layers} layers into ${S.pp} stage${S.pp > 1 ? 's' : ''} of ${L.layersPerStage}; activations pass between stages point to point.${bubble}</li>
      ${isMoE(m) ? `<li><b>EP</b> splits the ${m.ffn.experts} experts of each MoE layer over ${S.ep} GPU${S.ep > 1 ? 's' : ''} (${m.ffn.experts / Math.max(1, S.ep)} each); tokens travel to their experts and back (all-to-all), twice per MoE layer in each direction.</li>` : ''}
      <li><b>CP</b> splits each sequence into ${S.cp} chunk${S.cp > 1 ? 's' : ''}; attention passes key and value blocks around the ring of CP GPUs. Transformer Engine also offers an all-to-all variant that regroups attention heads instead, which needs the heads per GPU to be divisible by CP.</li></ul>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, S = layoutOf(m), L = trainingLayout(m, S), issues = validateLayout(m, S), lay = m.layouts.training;
    return `
      <p class="lead">When a model's weights, training state, activations or compute exceed one GPU, training is split across GPUs along up to five dimensions. Data parallelism (DP) splits the batch, tensor parallelism (TP) splits weight matrices, pipeline parallelism (PP) splits the layers, expert parallelism (EP) splits the experts, and context parallelism (CP) splits the sequence. Each dimension has to exchange specific data every step; this chapter counts those bytes.</p>
      <p>${esc(lay.text)}${lay.url ? ` (<a href="${esc(lay.url)}" target="_blank" rel="noreferrer">source</a>)` : ''}</p>
      ${controls(m, S)}
      <p class="status">${fmtN(L.N)} GPUs = DP ${S.dp} × TP ${S.tp} × PP ${S.pp} × CP ${S.cp}${isMoE(m) ? ` (EP ${S.ep} within them)` : ''}, ${fmtN(L.nodes)} node${L.nodes > 1 ? 's' : ''} of ${S.gpusPerNode}. ${fmtN(S.gbs)} sequences × ${fmtN(S.seq)} tokens = ${fmtParams(S.gbs * S.seq)} tokens per step, ${Number.isInteger(L.micro) ? `${fmtN(L.micro)} micro-batch${L.micro > 1 ? 'es' : ''} per DP rank` : ''}.</p>
      ${issues.length ? `<p class="issues">${issues.map(esc).join(' ')}</p>` : ''}
      <h2>GPUs and their coordinates</h2>${figNodes(m, S, W)}
      <h2>Memory per GPU</h2>${figMemory(m, S, ctx, W)}
      <h2>Bytes exchanged per step</h2>${figCollectives(m, S)}
      <h2>What each dimension splits</h2>${dimList(m, S, L)}
      ${measuredTable('parallel', m)}
      <p class="sources">Sources: <a href="https://arxiv.org/abs/1909.08053" target="_blank" rel="noreferrer">Megatron-LM (TP)</a> · <a href="https://arxiv.org/abs/2104.04473" target="_blank" rel="noreferrer">PP with 1F1B and interleaving</a> · <a href="https://arxiv.org/abs/2205.05198" target="_blank" rel="noreferrer">sequence parallelism and selective recomputation</a> · <a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">ZeRO</a> · <a href="https://arxiv.org/abs/2310.01889" target="_blank" rel="noreferrer">ring attention (CP)</a> · <a href="https://arxiv.org/abs/2006.16668" target="_blank" rel="noreferrer">GShard (EP)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    const S = layoutOf(ctx.m);
    root.querySelectorAll('[data-par]').forEach((el) => el.addEventListener('change', () => {
      const k = el.dataset.par, v = el.value;
      S[k] = /^\d+$/.test(v) ? Number(v) : v;      // numeric values keep 0 (ZeRO-0)
      if (k === 'gbs' && !(S.gbs >= 1)) S.gbs = 1;
      rerender();
    }));
    root.querySelector('#par-reset').addEventListener('click', (e) => { e.preventDefault(); delete layouts[ctx.m.id]; rerender(); });
    [['#bw-intra', (v) => { bwIntra = v; }], ['#bw-inter', (v) => { bwInter = v; }]].forEach(([id, set]) => root.querySelector(id).addEventListener('change', (e) => { set(e.target.value); rerender(); }));
  }

  (window.CHAPTERS = window.CHAPTERS || {}).parallel = { render, mount };
})();
