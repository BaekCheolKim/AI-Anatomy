// Chapter: Mixture of experts.
// Figures: (1) routing of example tokens in one MoE layer of the selected model, (2) total and
// per-token parameters of every model, (3) where the selected model's parameters are, (4) the
// weights split over the reader's GPUs: bytes per GPU and, for MoE, dispatch traffic.
// Figures are laid out at the chapter column's width (ctx.width) so that text keeps its size.
(function () {
  const TOKENS = ['The', ' cat', ' sat', ' on', ' the', ' mat', ' and', ' slept'];
  let sel = null;                // index of the selected example token; null shows all of them
  const dtypes = {};             // per model id: dtype overrides chosen in figure 4

  // Illustrative routing, seeded by model id. Respects E, k and group limits from config.json.
  function routes(m) {
    const f = m.ffn, rnd = mulberry32(hashStr(m.id));
    return TOKENS.map(() => {
      let pool = range(f.experts);
      if (f.groups) {
        const per = f.experts / f.groups;
        pool = shuffled(range(f.groups), rnd).slice(0, f.topkGroups).flatMap((g) => range(per).map((j) => g * per + j));
      }
      const experts = shuffled(pool, rnd).slice(0, f.topK).sort((a, b) => a - b);
      const raw = experts.map(() => 0.3 + rnd()), z = raw.reduce((s, x) => s + x, 0);
      return { experts, gates: raw.map((x) => x / z) };
    });
  }

  function routerSentence(m) {
    const f = m.ffn;
    if (f.router === 'sigmoid-grouped') {
      return `Each expert's score is sigmoid(x·w<sub>e</sub>). For selection only, a per-expert bias is added; after each training step it is lowered for overloaded experts and raised for underloaded ones (auxiliary-loss-free balancing), and a sequence-wise balance loss with weight 0.0001 is applied as well. Only the ${f.topkGroups} best-scoring of ${f.groups} groups of ${f.experts / f.groups} experts are eligible. The ${f.topK} selected scores are normalized to sum to 1 and multiplied by ${f.routedScale}.`;
    }
    if (f.router === 'topk-softmax') {
      return `The router, a linear layer with a bias, gives ${f.experts} logits. The ${f.topK} largest are kept, and a softmax over those ${f.topK} gives the weights.`;
    }
    const keep = f.normTopK ? `the ${f.topK} largest are kept and rescaled to sum to 1` : `the ${f.topK} largest select the experts`;
    const gate = f.sharedGate ? ' The shared expert\'s output is multiplied by sigmoid(x·w<sub>s</sub>), a gate with one weight vector.' : '';
    return `A softmax over the ${f.experts} router outputs gives probabilities; ${keep}.${gate}`;
  }

  function modelParagraph(m) {
    const P = params(m), f = m.ffn;
    if (!isMoE(m)) {
      const moeLinks = window.MODELS.filter(isMoE).map((x) => `<a href="#" data-model="${x.id}">${esc(x.name)}</a>`).join(', ');
      return `<p><b>${esc(m.name)}</b> is dense: each of its ${m.layers} layers has one FFN (${fmtN(m.hidden)} → ${fmtN(f.inter)} → ${fmtN(m.hidden)}) that every token uses, so all ${fmtParams(P.total)} parameters are used for every token. The figures below show it next to the MoE models: ${moeLinks}.</p>`;
    }
    const dense = f.denseLayers ? ` (layers 1–${f.denseLayers} use one dense FFN instead)` : '';
    const shared = f.shared ? ` and ${f.shared} shared expert` : ', and no shared expert,';
    const uses = f.shared ? `${f.topK} routed experts plus the shared expert` : `${f.topK} experts`;
    return `<p><b>${esc(m.name)}</b> has ${f.experts} routed experts${shared} in each of its ${P.moeLayers} MoE layers${dense}. Each token uses ${uses}, so ${fmtParams(P.active)} of its ${fmtParams(P.total)} parameters (${fmtPct(P.active, P.total)}) are used per token. ${routerSentence(m)}</p>`;
  }

  // ---- figure 1: one layer, token by token ----
  function figRouting(m, W) {
    const f = m.ffn, P = params(m);
    const tokRow = TOKENS.map((t, i) => `<button class="${sel === i ? 'on' : ''}" data-tok="${i}">${showTok(t)}</button>`).join('')
      + `<button class="${sel === null ? 'on' : ''}" data-tok="all">all 8</button>`;
    if (!isMoE(m)) {
      const label = `FFN ${fmtN(m.hidden)} → ${fmtN(f.inter)} → ${fmtN(m.hidden)}`;
      const body = `<rect x="0" y="0" width="${W}" height="44" class="seg0"/><text x="${W / 2}" y="27" text-anchor="middle" class="on-fill">${label}${W > 420 ? ': every token' : ''}</text>`;
      return `<div class="tokens">${tokRow}</div>${figure(body, W, 46, 'dense FFN')}
        <p class="caption">One layer of ${esc(m.name)}: a single FFN with ${fmtParams(P.denseFFN / m.layers)} parameters processes every token.</p>`;
    }
    const narrow = W < 600, E = f.experts, cols = narrow ? 16 : 32;
    const left = f.groups ? (narrow ? 26 : 62) : 0, right = f.shared ? 56 : 0;
    const step = Math.max(8, Math.floor((W - left - right) / cols)), cell = step - (step > 12 ? 3 : 2);
    const R = routes(m), counts = new Array(E).fill(0);
    R.forEach((r) => r.experts.forEach((e) => { counts[e]++; }));
    const chosen = sel === null ? null : R[sel];
    let s = '';
    for (let e = 0; e < E; e++) {
      const x = left + (e % cols) * step, y = Math.floor(e / cols) * step;
      const i = chosen ? chosen.experts.indexOf(e) : -1;
      const c = chosen ? (i >= 0 ? 1 : 0) : Math.min(counts[e], 2);
      const tip = chosen ? `expert ${e}${i >= 0 ? `: weight ${chosen.gates[i].toFixed(2)}` : ''}` : `expert ${e}: selected by ${counts[e]} of 8 tokens`;
      s += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" class="ex c${c}"><title>${tip}</title></rect>`;
    }
    if (f.groups) {
      const rowsPerGroup = f.experts / f.groups / cols;
      for (let g = 0; g < f.groups; g++) s += `<text x="0" y="${g * rowsPerGroup * step + cell - 3}" class="axis">${narrow ? `g${g + 1}` : `group ${g + 1}`}</text>`;
    }
    if (f.shared) {
      const sx = left + cols * step + 14;
      s += `<rect x="${sx}" y="0" width="${cell}" height="${cell}" class="ex c1"><title>shared expert: every token</title></rect><text x="${sx}" y="${cell + 13}" class="axis">shared</text>`;
    }
    const h = Math.max(Math.ceil(E / cols) * step, f.shared ? cell + 16 : 0);
    const used = counts.filter((c) => c > 0).length;
    const exp = (n) => expertsTouched(E, f.topK, n).toFixed(n < 64 ? 1 : 0);
    const status = chosen
      ? `<b>${showTok(TOKENS[sel])}</b> → experts ${chosen.experts.join(', ')}${f.shared ? ' and the shared expert' : ''}; weights ${chosen.gates.map((g) => g.toFixed(2)).join(', ')}${f.routedScale ? `, each then × ${f.routedScale}` : ''}.`
      : `These 8 tokens select ${used} different experts of ${E}. With uniform selection, n tokens select E·(1 − (1 − k/E)<sup>n</sup>) different experts on average: ${exp(8)} for 8 tokens, ${exp(64)} for 64, ${exp(512)} for 512. In a decode step, every selected expert's weights are read from memory.`;
    return `<div class="tokens">${tokRow}</div>${figure(s, W, h, 'experts of one MoE layer')}
      <p class="status">${status}</p>
      <p class="caption">One MoE layer of ${esc(m.name)}. Each square is one expert, an FFN with ${fmtParams(P.perExpert)} parameters${chosen ? '; filled: selected for this token' : '; filled: selected by one example token, darker: by two or more'}. The page doesn't run the model, so which experts are selected is illustrative. E, k${f.groups ? ', the group limit' : ''} and the sizes come from config.json.</p>`;
  }

  // ---- figure 2: total and per-token parameters, every model ----
  function figTotals(m, W) {
    const models = window.MODELS.slice().sort((a, b) => params(a).total - params(b).total);
    const narrow = W < 560, rowH = narrow ? 44 : 30, top = 26;
    const x0 = narrow ? 8 : 150, x1 = narrow ? W - 12 : W - 230, sx = logScale(1e9, 1e12, x0, x1);
    let s = '';
    [1e9, 1e10, 1e11, 1e12].forEach((t) => {
      s += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${top - 8}" y2="${top + models.length * rowH - 6}" class="grid"/><text x="${sx(t)}" y="${top - 14}" text-anchor="middle" class="axis">${fmtParams(t)}</text>`;
    });
    models.forEach((mm, i) => {
      const P = params(mm), on = mm.id === m.id;
      const yName = top + i * rowH + 12, y = narrow ? yName + 16 : yName - 4;
      const vals = `${fmtParams(P.total)} · ${fmtParams(P.active)} per token${isMoE(mm) ? ` (${fmtPct(P.active, P.total)})` : ''}`;
      s += `<g class="row${on ? ' on' : ''}" data-model="${mm.id}"><title>${esc(mm.name)}: ${fmtN(P.total)} parameters, ${fmtN(P.active)} used per token</title>`
        + `<rect x="0" y="${top + i * rowH - 4}" width="${W}" height="${rowH - 2}" class="hit"/>`
        + `<text x="0" y="${yName}" class="name">${esc(mm.name)}</text>`
        + (isMoE(mm) ? `<line x1="${sx(P.active)}" x2="${sx(P.total)}" y1="${y}" y2="${y}" class="span"/>` : '')
        + `<circle cx="${sx(P.total)}" cy="${y}" r="5.5" class="tot"/><circle cx="${sx(P.active)}" cy="${y}" r="3.5" class="act"/>`
        + `<text x="${narrow ? W : x1 + 22}" y="${yName}"${narrow ? ' text-anchor="end"' : ''} class="val">${vals}</text></g>`;
    });
    return `${figure(s, W, top + models.length * rowH, 'total and per-token parameters')}
      <p class="caption">Ring: total parameters. Dot: parameters used per token. Log scale. In a dense model the two coincide. Click a row to select that model.</p>`;
  }

  // ---- figure 3: where the selected model's parameters are ----
  function figBreakdown(m, W) {
    const P = params(m), activeRouted = P.active - (P.language - P.routed);
    // [name, stored, used per token, shade]: routed experts light, everything else dark.
    const parts = [
      ['Embedding table', P.embed + P.pos, P.embed + P.pos, 0],
      ['Attention', P.attn, P.attn, 0],
      ['Dense FFN', P.denseFFN, P.denseFFN, 0],
      ['Router', P.router, P.router, 0],
      ['Shared experts', P.shared, P.shared, 0],
      ['Routed experts', P.routed, activeRouted, 1],
      ['Norms', P.norms + P.finalNorm, P.norms + P.finalNorm, 0],
      ['Output layer', P.head, P.head, 0],
      ['Vision encoder', P.vision, 0, 0],
    ].filter(([, stored]) => stored > 0);
    const k = W / P.total, barH = 22;
    const bar = (y, col) => {
      let x = 0, out = '';
      parts.forEach(([name, stored, used, shade]) => {
        const v = col === 1 ? stored : used, w = v * k;
        if (w > 0) out += `<rect x="${x}" y="${y}" width="${Math.max(w, 0.5)}" height="${barH}" class="seg${shade} sep"><title>${name}: ${fmtN(v)}</title></rect>`;
        x += w;
      });
      return out;
    };
    const s = `<text x="0" y="12" class="axis">stored: ${fmtParams(P.total)}</text>${bar(18, 1)}`
      + `<text x="0" y="66" class="axis">used per text token: ${fmtParams(P.active)}</text>${bar(72, 2)}`;
    const rows = parts.map(([name, stored, used]) => `<tr><td>${name}</td><td class="r">${fmtParams(stored)}</td><td class="r">${fmtParams(used)}</td><td class="r">${fmtPct(stored, P.total)}</td></tr>`).join('');
    return `${figure(s, W, 96, 'parameters by component')}
      <table class="data"><thead><tr><th>component</th><th class="r">stored</th><th class="r">used per token</th><th class="r">share of stored</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="caption">${esc(m.name)}: parameters by component, stored (top) and used for one text token (bottom), on the same scale.${isMoE(m) ? ' Light: routed experts. Dark: everything else.' : ''} The embedding table is counted in full in both bars, although each token reads one row of it; most authors count it as active, OpenAI's gpt-oss count leaves it out.${P.mtp ? ` Not shown: the multi-token-prediction module, ${fmtParams(P.mtp)} parameters, used only for training and speculative decoding.` : ''}</p>`;
  }

  // ---- figure 4: weights over the reader's GPUs ----
  function figGPUs(m, ctx, W) {
    const n = ctx.gpus, mem = ctx.mem;
    let d = { ...releaseDtypes(m), ...(dtypes[m.id] || {}) };
    if (!isMoE(m)) d = Object.fromEntries(Object.keys(d).map((k) => [k, d.linear]));   // dense: one dtype for all weights
    const Wt = weightBytes(m, d), expertsB = Wt.byClass.experts, otherB = Wt.total - expertsB;
    const opts = m.release?.all === 'fp32' ? ['fp32', 'bf16', 'fp8'] : ['bf16', 'fp8'];
    const minGPUs = Math.ceil(Wt.total / mem);
    const controls = isMoE(m)
      ? `<p class="controls">Routed experts in ${dtypeSelect('experts', d.experts, ['bf16', 'fp8', 'mxfp4'])} · attention, shared experts and dense FFNs in ${dtypeSelect('linear', d.linear, opts)} · embedding, norms and routers stay in ${DTYPE_NAME[d.embed]}</p>`
      : `<p class="controls">Weights in ${dtypeSelect('linear', d.linear, opts)}</p>`;
    let perGPU, lines, segs;
    if (!isMoE(m)) {
      perGPU = Wt.total / n;
      segs = [[perGPU, 'seg0', 'weights']];
      lines = `<p>${esc(m.name)} has no experts. Split over ${n} GPU${n > 1 ? 's' : ''} with tensor parallelism (every weight matrix divided into ${n} parts), each GPU holds ${fmtBytes(perGPU)} of the ${fmtBytes(Wt.total)} of weights.</p>`;
    } else {
      const f = m.ffn, P = params(m), E = f.experts, perLayer = Math.ceil(E / n), even = E % n === 0;
      perGPU = otherB + expertsB * (perLayer / E);
      segs = [[otherB, 'seg0', 'other weights (a copy on each GPU)'], [expertsB * (perLayer / E), 'seg1', `routed experts: ${perLayer} per layer`]];
      const msg = dispatchBytes(m) * 2;
      lines = (n === 1
        ? `<p>On one GPU, all ${E} experts of every MoE layer are local: ${fmtN(E * P.moeLayers)} expert FFNs, ${fmtBytes(expertsB)}, plus all other weights, ${fmtBytes(otherB)}.</p>`
        : `<p>${even ? `Split over ${n} GPUs, each GPU holds ${perLayer} of the ${E} experts of every MoE layer` : `${E} experts don't divide evenly over ${n} GPUs; the fullest GPU holds ${perLayer} experts of every MoE layer`}: ${fmtN(perLayer * P.moeLayers)} expert FFNs, ${fmtBytes(expertsB * perLayer / E)}. All other weights, ${fmtBytes(otherB)}, are copied to each GPU.</p>`)
        + `<p>${n > 1 ? `Dispatch: each token's hidden vector (${fmtN(m.hidden)} values, ${fmtBytes(m.hidden * 2)} in BF16) is sent to the GPUs holding its ${f.topK} experts, and ${f.topK} outputs of the same size return: ${fmtBytes(msg)} per token per MoE layer, ${fmtBytes(msg * P.moeLayers)} per token for one forward pass through ${P.moeLayers} MoE layers. With uniform selection, ${n - 1} of every ${n} copies go to another GPU.` : 'With one GPU every expert is local, and there is no dispatch traffic.'}</p>`;
    }
    const scale = (W - 4) / Math.max(perGPU, mem) / 1.04;
    let x = 0, s = '';
    segs.forEach(([v, cls, name]) => { s += `<rect x="${x}" y="22" width="${Math.max(v * scale, 0.5)}" height="24" class="${cls} sep"><title>${name}: ${fmtBytes(v)}</title></rect>`; x += v * scale; });
    const mx = mem * scale, fits = perGPU <= mem, anchor = mx > W - 70 ? 'end' : mx < 70 ? 'start' : 'middle';
    s += `<line x1="${mx}" x2="${mx}" y1="12" y2="56" class="memline"/><text x="${mx}" y="8" text-anchor="${anchor}" class="axis">GPU memory ${fmtBytes(mem)}</text>`;
    s += `<text x="0" y="72" class="val">${fmtBytes(perGPU)} per GPU · ${fits ? `${fmtBytes(mem - perGPU)} left` : `${fmtBytes(perGPU - mem)} over`}</text>`;
    return `${controls}${figure(s, W, 78, 'weights per GPU')}${lines}
      <p>All weights at these precisions: ${fmtBytes(Wt.total)}${isMoE(m) ? `, ${fmtBytes(expertsB)} of them routed experts` : ''}. Holding one copy needs at least ${minGPUs} GPU${minGPUs > 1 ? 's' : ''} of ${fmtBytes(mem)}, before KV cache, activations and runtime buffers.${fits ? ` Here ${fmtBytes(mem - perGPU)} per GPU is left for those.` : ''}</p>
      <p class="caption">GPU memory and the number of GPUs are set at the top of the page. ${isMoE(m) ? 'Assumed layout: routed experts split evenly over the GPUs (expert parallelism); every other weight copied to each GPU, as with data-parallel attention. ' : ''}Default precisions are those of the published weights.</p>`;
  }

  // ---- formulas ----
  function details(m) {
    const P = params(m), f = m.ffn, H = m.hidden;
    const parts = [P.vision ? `vision encoder ${fmtN(P.vision)}` : '', P.mtp ? `multi-token prediction ${fmtN(P.mtp)}` : ''].filter(Boolean);
    const check = `Check: computed ${fmtN(P.checkpoint)} parameters in the checkpoint${parts.length ? ` (including ${parts.join(' and ')})` : ''}; Hugging Face reports ${fmtN(m.checkpoint.params)}${m.checkpoint.buffers ? `, which includes ${fmtN(m.checkpoint.buffers)} values of non-parameter buffers` : ''}. Published: “${esc(m.published.text)}”.${m.published.convention ? ` ${esc(m.published.convention)}` : ''}`;
    if (!isMoE(m)) {
      return `<ul class="formulas">
        <li>FFN: y = ${f.gated ? '(SiLU(x·W<sub>gate</sub>) ⊙ x·W<sub>up</sub>)·W<sub>down</sub>' : 'GELU(x·W<sub>1</sub> + b<sub>1</sub>)·W<sub>2</sub> + b<sub>2</sub>'}, with ${f.gated ? 'W<sub>gate</sub>, W<sub>up</sub>' : 'W<sub>1</sub>'} ∈ ℝ<sup>${fmtN(H)} × ${fmtN(f.inter)}</sup>: ${fmtN(P.denseFFN / m.layers)} parameters per layer.</li>
        <li>FLOPs per token: 2 × ${fmtParams(P.matmul)} weights multiplied = ${fmtFlops(2 * P.matmul)}, plus attention over the context.</li>
        <li>${check}</li></ul>`;
    }
    const ml = moeLayer(m), Fe = f.expertInter;
    const expert = f.swigluLimit
      ? `y<sub>e</sub> = ((a ⊙ σ(1.702·a)) ⊙ (u + 1))·W<sub>down,e</sub> + b<sub>down,e</sub>, with a = min(x·W<sub>gate,e</sub> + b<sub>gate,e</sub>, ${f.swigluLimit}) and u = clamp(x·W<sub>up,e</sub> + b<sub>up,e</sub>, −${f.swigluLimit}, ${f.swigluLimit}) (swiglu_limit in config.json; OpenAI's reference code)`
      : 'y<sub>e</sub> = (SiLU(x·W<sub>gate,e</sub>) ⊙ x·W<sub>up,e</sub>)·W<sub>down,e</sub>';
    const flopsTok = 2 * (f.topK + f.shared) * 3 * H * Fe + 2 * H * f.experts;
    return `<ul class="formulas">
      <li>Router: ${f.router === 'sigmoid-grouped' ? 's = σ(x·W<sub>r</sub>)' : f.router === 'topk-softmax' ? 's = x·W<sub>r</sub> + b' : 's = x·W<sub>r</sub>'}, W<sub>r</sub> ∈ ℝ<sup>H × E</sup> = ℝ<sup>${fmtN(H)} × ${f.experts}</sup>. ${routerSentence(m)}</li>
      <li>Expert e: ${expert}, with W<sub>gate,e</sub>, W<sub>up,e</sub> ∈ ℝ<sup>${fmtN(H)} × ${fmtN(Fe)}</sup> and W<sub>down,e</sub> ∈ ℝ<sup>${fmtN(Fe)} × ${fmtN(H)}</sup>: ${fmtN(ml.perExpert)} parameters.</li>
      <li>Layer output: y = Σ<sub>e ∈ selected</sub> g<sub>e</sub>·y<sub>e</sub>${f.shared ? (f.sharedGate ? ' + σ(x·w<sub>s</sub>)·shared(x)' : ' + shared(x)') : ''}.</li>
      <li>Per MoE layer: ${f.experts} × ${fmtN(ml.perExpert)} + router ${fmtN(ml.router)}${f.shared ? ` + shared ${fmtN(ml.shared)}` : ''} = ${fmtN(ml.total)} parameters; ${fmtN(ml.active)} used per token.</li>
      <li>FLOPs per token per MoE layer: 2·(k${f.shared ? ' + shared' : ''})·3·H·F<sub>e</sub> + 2·H·E = ${fmtFlops(flopsTok)}. Whole forward pass: about 2 × ${fmtParams(P.matmul)} = ${fmtFlops(2 * P.matmul)} per token, plus attention over the context.</li>
      <li>Experts selected by n tokens, each selecting k of E uniformly: E·(1 − (1 − k/E)<sup>n</sup>).</li>
      <li>${check}</li></ul>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    const src = [`${configLink(m)}`];
    if (m.published.url) src.push(`<a href="${esc(m.published.url)}" target="_blank" rel="noreferrer">published parameter counts</a>`);
    if (m.ffn.router === 'sigmoid-grouped') src.push('<a href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noreferrer">DeepSeek-V3 report, routing and load balancing</a>');
    return `
      <p class="lead">In a dense model, every token passes through the same feed-forward network (FFN) in each layer. A mixture-of-experts (MoE) layer stores E separate FFNs, called experts, and a router: a linear layer that computes one score per expert for each token. Each token is processed only by the k experts with the highest scores, and their outputs are summed with weights from the router. Stored parameters grow with E; computation per token grows with k.</p>
      ${modelParagraph(m)}
      <h2>One layer, token by token</h2>${figRouting(m, W)}
      <h2>Total and per-token parameters</h2>${figTotals(m, W)}
      <h2>Where the parameters are</h2>${figBreakdown(m, W)}
      <h2>${isMoE(m) ? 'Experts' : 'Weights'} on GPUs</h2>${figGPUs(m, ctx, W)}
      <details class="more"><summary>Formulas and shapes</summary>${details(m)}</details>
      <p class="sources">Sources: ${src.join(' · ')}.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelectorAll('[data-tok]').forEach((b) => b.addEventListener('click', () => {
      sel = b.dataset.tok === 'all' ? null : Number(b.dataset.tok);
      rerender();
    }));
    root.querySelectorAll('select[data-dtype]').forEach((el) => el.addEventListener('change', () => {
      dtypes[ctx.m.id] = { ...(dtypes[ctx.m.id] || {}), [el.dataset.dtype]: el.value };
      rerender();
    }));
  }

  (window.CHAPTERS = window.CHAPTERS || {}).moe = { render, mount };
})();
