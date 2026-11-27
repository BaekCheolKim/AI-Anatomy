// Chapter: One RL step on GPUs.
// Figures: (1) FLOPs of the phases of one GRPO step; (2) GPU memory per phase when generation and
// training share the GPUs (colocated), with the bytes moved between phases.
// A model's published RL recipe (models.js layouts.rl) sets the defaults; otherwise example values.
(function () {
  const EXAMPLE = { prompts: 32, genPrompts: 32, G: 8, prompt: 2048, answer: 4096, kl: true };
  const state = {};   // per model id
  const PROMPTS = [256, 512, 2048, 8192], ANSWERS = [512, 1024, 4096, 6144, 16384];
  // Lengths that fit the model's context: prompt at most half of it, prompt + answer within it.
  const promptOpts = (m) => { const o = PROMPTS.filter((v) => v <= m.context / 2); return o.length ? o : [Math.floor(m.context / 2)]; };
  const answerOpts = (m, p) => { const o = ANSWERS.filter((v) => p + v <= m.context); return o.length ? o : [m.context - p]; };
  function clampLengths(m, W8) {
    const po = promptOpts(m);
    if (!po.includes(W8.prompt)) W8.prompt = po.filter((v) => v <= W8.prompt).pop() ?? po[0];
    const ao = answerOpts(m, W8.prompt);
    if (!ao.includes(W8.answer)) W8.answer = ao.filter((v) => v <= W8.answer).pop() ?? ao[0];
  }

  const settings = (m) => {
    if (state[m.id]) return state[m.id];
    const r = m.layouts.rl;
    state[m.id] = r ? { prompts: r.prompts, genPrompts: r.genPrompts, G: r.G, prompt: r.prompt, answer: r.total - r.prompt, kl: r.kl > 0 } : { ...EXAMPLE };
    return state[m.id];
  };

  function phases(m, W8) {
    const P = params(m), nGen = W8.genPrompts * W8.G, nTrain = W8.prompts * W8.G, ctxAvg = W8.prompt + W8.answer / 2, seqLen = W8.prompt + W8.answer;
    const gen = nGen * W8.prompt * forwardFlopsPerToken(m, W8.prompt, 'causal').total + nGen * W8.answer * forwardFlopsPerToken(m, ctxAvg, 'decode').total;
    const fwd = nTrain * seqLen * forwardFlopsPerToken(m, seqLen, 'causal').total;
    return [
      ['Generate', gen, `${fmtN(nGen)} answers (${W8.genPrompts} prompts × G ${W8.G}) of up to ${fmtN(W8.answer)} tokens, by an inference engine${W8.genPrompts > W8.prompts ? `; groups whose rewards are all equal are dropped, and ${W8.prompts} of the remaining prompts are trained on (dynamic sampling)` : ''}`, 'seg-out'],
      ['Score', 0, 'rule-based checker on the CPU (no model FLOPs)', ''],
      ['Old-policy log-probabilities', fwd, `one forward pass over the ${fmtN(nTrain)} answers used for training`, 'a'],
      ['Reference log-probabilities', W8.kl ? fwd : 0, W8.kl ? 'one forward pass of the frozen reference model, for the KL term' : 'skipped: the KL coefficient is 0, so no reference model is loaded', 'a'],
      ['Train', 3 * fwd, 'forward and backward over the answers, then the optimizer step', 'g'],
      ['Copy weights to the inference engine', 0, `${fmtBytes(P.total * 2)} of BF16 weights in total`, ''],
    ];
  }

  function figPhases(m, W8, W) {
    const ph = phases(m, W8), max = Math.max(...ph.map((p) => p[1])), lw = W < 560 ? 0 : 250, bw = W - lw - (W < 560 ? 0 : 100), rowH = W < 560 ? 44 : 30;
    let s = '';
    ph.forEach(([name, f, , cls], i) => {
      const y = i * rowH + (W < 560 ? 14 : 2);
      s += `<text x="0" y="${W < 560 ? y : y + 14}" class="name">${i + 1} ${name}</text>${f ? `<rect x="${lw}" y="${W < 560 ? y + 4 : y + 2}" width="${Math.max(1, (f / max) * bw)}" height="16" class="${cls}"/>` : ''}`
        + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 14}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${f ? fmtFlops(f) : '—'}</text>`;
    });
    const sel = (k, opts) => `<select data-rl="${k}">${opts.map((v) => `<option${v === W8[k] ? ' selected' : ''}>${v}</option>`).join('')}</select>`;
    return `<p class="controls"><label class="nw">prompts trained ${sel('prompts', [8, 32, 64, 128, 512])}</label> <label class="nw">prompts generated ${sel('genPrompts', [8, 32, 64, 128, 512])}</label> <label class="nw">G ${sel('G', [4, 8, 16])}</label> <label class="nw">prompt tokens ${sel('prompt', promptOpts(m))}</label> <label class="nw">answer tokens ${sel('answer', answerOpts(m, W8.prompt))}</label> <label class="nw"><input type="checkbox" id="rl-kl"${W8.kl ? ' checked' : ''}> KL term (reference model)</label></p>
      ${figure(s, W, ph.length * rowH + (W < 560 ? 14 : 4), 'phases of one RL step')}
      <table class="data"><tbody>${ph.map(([name, , what], i) => `<tr><td>${i + 1} ${name}</td><td class="f">${what}</td></tr>`).join('')}</tbody></table>
      <p class="caption">FLOPs of one GRPO step for ${esc(m.name)}, all answers at full length. Generation has the fewest FLOPs per token but usually takes most of the time: each decode step reads all the weights and the KV cache to produce one token per answer, so it is limited by memory bandwidth rather than by arithmetic (Prefill and decode).</p>`;
  }

  function figMemory(m, W8, ctx, W) {
    const S = { optimizer: 'adamw', ...m.layouts.training, seq: W8.prompt + W8.answer }, L = trainingLayout(m, S), n = L.N, r = m.layouts.rl, P = params(m);
    const train = trainingMemoryPerGPU(m, S), wTrain = train[0].bytes;
    const state2 = train.filter((x) => x.kind !== 'a' && x.name !== 'Weights, BF16').reduce((s, x) => s + x.bytes, 0);
    const act = train.filter((x) => x.kind === 'a').reduce((s, x) => s + x.bytes, 0);
    // Inference engine: the recipe's layout, else TP × PP of the training layout. With EP, the routed experts are split
    // over the EP ranks and the other weights over TP; each TP group (a data-parallel attention rank) holds its own sequences.
    const g = r?.gen ?? { tp: S.tp, pp: S.pp }, tp = g.tp, pp = g.pp ?? 1, ep = g.ep ?? 0, engine = Math.max(tp * pp, ep), engines = Math.max(1, Math.floor(n / engine));
    const wInf = 2 * (ep ? (P.total - P.routed) / tp + P.routed / ep : P.total / (tp * pp));
    const seqPerGroup = (W8.genPrompts * W8.G * tp * pp) / n, kv = (seqPerGroup * cacheBytesPerGPU(m, W8.prompt + W8.answer, tp)) / pp;
    const bars = [
      ['generation', [['w', 'weights for the inference engine, BF16', wInf], ['kv', 'KV cache of all answers', kv]]],
      ['training', [['w', 'weights', wTrain], ['g', 'gradients, master weights, optimizer state', state2], ['a', 'activations', act]]],
    ];
    // Narrow screens: each bar's total follows its name, and the memory line is labelled below the bars.
    const narrow = W < 560, max = Math.max(ctx.mem, ...bars.map(([, segs]) => segs.reduce((s, x) => s + x[2], 0))), lw = narrow ? 0 : 110, bw = W - lw - (narrow ? 2 : 80);
    let s = '';
    bars.forEach(([name, segs], i) => {
      const y = i * 40 + 18, total = fmtBytes(segs.reduce((t, z) => t + z[2], 0)); let x = lw;
      s += narrow ? `<text x="0" y="${y - 4}" class="name">${name}: ${total}</text>` : `<text x="0" y="${y + 15}" class="name">${name}</text><text x="${lw + bw + 8}" y="${y + 15}" class="val">${total}</text>`;
      segs.forEach(([k, label, v]) => { if (v > 0) { s += `<rect x="${x}" y="${y}" width="${Math.max(0.5, (v / max) * bw)}" height="22" class="${k === 'kv' ? 'kvbar' : k} sep"><title>${label}: ${fmtBytes(v)}</title></rect>`; x += (v / max) * bw; } });
    });
    const mx = lw + (ctx.mem / max) * bw, yb = bars.length * 40 + 16;
    s += `<line x1="${mx}" x2="${mx}" y1="8" y2="${yb}" class="memline"/>` + (narrow
      ? `<text x="${mx}" y="${yb + 12}" text-anchor="${mx > W / 2 ? 'end' : 'start'}" class="axis">GPU memory ${fmtBytes(ctx.mem)}</text>`
      : `<text x="${mx}" y="6" text-anchor="middle" class="axis">GPU memory ${fmtBytes(ctx.mem)}</text>`);
    const genText = ep ? `${engines} inference engine${engines > 1 ? 's' : ''} of ${engine} GPUs (TP ${tp}, EP ${ep}: routed experts split over ${ep} GPUs, the other weights over ${tp})` : `${engines} inference engine${engines > 1 ? 's' : ''} of TP × PP = ${engine} GPU${engine > 1 ? 's' : ''}`;
    return `${figure(s, W, narrow ? yb + 16 : bars.length * 40 + 22, 'memory per phase')}
      <p class="status">Per GPU, with the ${fmtN(n)} GPUs of ${esc(m.name)}'s training layout (TP ${S.tp}, PP ${S.pp}${isMoE(m) ? `, EP ${S.ep}` : ''}, DP ${S.dp}) and ${genText}, each engine holding one copy of the weights. The KV cache is that of all answers at full length. Moving the trainer's weights and state to host memory and back costs ${fmtBytes(2 * (wTrain + state2))} per GPU per step over the CPU–GPU link; copying the updated weights into the inference engine writes ${fmtBytes(wInf)} per GPU.</p>
      <p class="caption">Colocated RL: the inference engine and the trainer take turns on the same GPUs. In NeMo-RL, before the weight copy the trainer moves its gradients and optimizer state to host memory; after the copy it also moves its BF16 weights there, and the inference engine then allocates the KV cache of all answers. Before training, the engine releases its KV cache and weights and the trainer state returns to GPU memory. HybridFlow (verl) instead reuses the training weight shards for generation. In a disaggregated setup the two run on separate GPUs, and the weight copy and the generated sequences cross between them. Activations are approximate.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, W8 = settings(m), r = m.layouts.rl;
    clampLengths(m, W8);
    const recipe = r ? ` The defaults follow ${esc(r.text)} (<a href="${esc(r.url)}" target="_blank" rel="noreferrer">recipe</a>).` : ' The defaults are example values.';
    return `
      <p class="lead">One RL step alternates between inference and training. An inference engine generates G answers for each prompt; a checker scores them; forward passes compute the token probabilities under the policy that generated them and, when the objective has a KL term, under a frozen reference model; the trainer runs forward and backward passes over the answers and updates the weights; the new weights are copied into the inference engine for the next step.${recipe}</p>
      <h2>The phases of one step</h2>${figPhases(m, W8, W)}
      <h2>GPU memory in each phase</h2>${figMemory(m, W8, ctx, W)}
      <p class="sources">Sources: <a href="https://github.com/NVIDIA-NeMo/RL" target="_blank" rel="noreferrer">NeMo-RL</a> · <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">GRPO</a> · <a href="https://arxiv.org/abs/2409.19256" target="_blank" rel="noreferrer">HybridFlow (verl)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    const W8 = settings(ctx.m);
    root.querySelectorAll('[data-rl]').forEach((el) => el.addEventListener('change', () => {
      const k = el.dataset.rl, v = Number(el.value);
      W8[k] = v;
      if (k === 'prompts' && W8.genPrompts < v) W8.genPrompts = v;   // at least as many prompts generated as trained
      if (k === 'genPrompts' && W8.prompts > v) W8.prompts = v;
      rerender();
    }));
    root.querySelector('#rl-kl').addEventListener('change', (e) => { W8.kl = e.target.checked; rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {})['rl-gpus'] = { render, mount };
})();
