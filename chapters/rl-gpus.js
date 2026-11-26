// Chapter: One RL step on GPUs.
// Figures: (1) FLOPs of the phases of one GRPO step; (2) GPU memory per phase when generation and
// training share the GPUs (colocated), with the bytes moved between phases.
(function () {
  const S0 = { prompts: 32, G: 8, prompt: 2048, answer: 4096 };
  let W8 = { ...S0 };

  function phases(m) {
    const P = params(m), n = W8.prompts * W8.G, toks = n * W8.answer, ctxAvg = W8.prompt + W8.answer / 2;
    const gen = n * W8.prompt * forwardFlopsPerToken(m, W8.prompt, 'causal').total + toks * forwardFlopsPerToken(m, ctxAvg, 'decode').total;
    const seqLen = W8.prompt + W8.answer, fwd = n * seqLen * forwardFlopsPerToken(m, seqLen, 'causal').total;
    return [
      ['Generate', gen, `${fmtN(n)} answers (${W8.prompts} prompts × G ${W8.G}) of up to ${fmtN(W8.answer)} tokens, by an inference engine`],
      ['Score', 0, 'rule-based checker on the CPU (no model FLOPs)'],
      ['Old and reference log-probabilities', 2 * fwd, 'two forward passes over all answers: the generating policy and the frozen reference model'],
      ['Train', 3 * fwd, 'forward and backward over all answers, then the optimizer step'],
      ['Copy weights to the inference engine', 0, `${fmtBytes(P.total * 2)} of BF16 weights in total`],
    ];
  }

  function figPhases(m, W) {
    const ph = phases(m), max = Math.max(...ph.map((p) => p[1])), lw = W < 560 ? 0 : 250, bw = W - lw - (W < 560 ? 0 : 100), rowH = W < 560 ? 44 : 30;
    let s = '';
    ph.forEach(([name, f], i) => {
      const y = i * rowH + (W < 560 ? 14 : 2);
      s += `<text x="0" y="${W < 560 ? y : y + 14}" class="name">${i + 1} ${name}</text>${f ? `<rect x="${lw}" y="${W < 560 ? y + 4 : y + 2}" width="${Math.max(1, (f / max) * bw)}" height="16" class="${i === 0 ? 'seg-out' : i === 2 ? 'a' : 'g'}"/>` : ''}`
        + `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 14}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${f ? fmtFlops(f) : '—'}</text>`;
    });
    return `<p class="controls"><label class="nw">prompts <select data-rl="prompts">${[8, 32, 128, 512].map((v) => `<option${v === W8.prompts ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">G <select data-rl="G">${[4, 8, 16].map((v) => `<option${v === W8.G ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">prompt tokens <select data-rl="prompt">${[512, 2048, 8192].map((v) => `<option${v === W8.prompt ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">answer tokens <select data-rl="answer">${[1024, 4096, 6144, 16384].map((v) => `<option${v === W8.answer ? ' selected' : ''}>${v}</option>`).join('')}</select></label></p>
      ${figure(s, W, ph.length * rowH + (W < 560 ? 14 : 4), 'phases of one RL step')}
      <table class="data"><tbody>${ph.map(([name, , what], i) => `<tr><td>${i + 1} ${name}</td><td class="f">${what}</td></tr>`).join('')}</tbody></table>
      <p class="caption">FLOPs of one GRPO step for ${esc(m.name)}, all answers at full length. Generation has the fewest FLOPs per token but usually takes most of the time: each decode step reads all the weights and the KV cache to produce one token per answer, so it is limited by memory bandwidth rather than by arithmetic (Prefill and decode).</p>`;
  }

  function figMemory(m, ctx, W) {
    const S = { optimizer: 'adamw', ...m.layouts.training, seq: W8.prompt + W8.answer }, L = trainingLayout(m, S), n = L.N;
    const train = trainingMemoryPerGPU(m, S), state = train.filter((r) => r.kind !== 'a' && r.name !== 'Weights, BF16').reduce((s, r) => s + r.bytes, 0);
    const engine = S.tp * S.pp, engines = Math.max(1, Math.floor(n / engine));   // one inference engine per model-parallel group
    const wInf = (params(m).total * 2) / engine, kv = (W8.prompts * W8.G * (W8.prompt + W8.answer) * kvBytesPerToken(m)) / n;
    const act = train.filter((r) => r.kind === 'a').reduce((s, r) => s + r.bytes, 0), wTrain = train[0].bytes;
    const bars = [
      ['generation', [['w', 'weights for the inference engine, BF16', wInf], ['kv', 'KV cache of all answers', kv], ['g-mid', 'training state, moved to host memory', 0]]],
      ['training', [['w', 'weights', wTrain], ['g', 'gradients, master weights, optimizer state', state], ['a', 'activations', act]]],
    ];
    const max = Math.max(ctx.mem, ...bars.map(([, segs]) => segs.reduce((s, x) => s + x[2], 0))), lw = W < 560 ? 0 : 110, bw = W - lw - (W < 560 ? 0 : 80);
    let s = '';
    bars.forEach(([name, segs], i) => {
      const y = i * 40 + 18; let x = lw;
      s += `<text x="0" y="${W < 560 ? y - 4 : y + 15}" class="name">${name}</text>`;
      segs.forEach(([k, label, v]) => { if (v > 0) { s += `<rect x="${x}" y="${y}" width="${Math.max(0.5, (v / max) * bw)}" height="22" class="${k === 'kv' ? 'kvbar' : k} sep"><title>${label}: ${fmtBytes(v)}</title></rect>`; x += (v / max) * bw; } });
      s += `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y - 4 : y + 15}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${fmtBytes(segs.reduce((t, z) => t + z[2], 0))}</text>`;
    });
    const mx = lw + (ctx.mem / max) * bw;
    s += `<line x1="${mx}" x2="${mx}" y1="8" y2="${bars.length * 40 + 16}" class="memline"/><text x="${mx}" y="6" text-anchor="middle" class="axis">GPU memory ${fmtBytes(ctx.mem)}</text>`;
    return `${figure(s, W, bars.length * 40 + 22, 'memory per phase')}
      <p class="status">Per GPU, with the ${fmtN(n)} GPUs of ${esc(m.name)}'s training layout (${m.layouts.training.example ? 'example' : 'published'}: TP ${S.tp}, PP ${S.pp}${isMoE(m) ? `, EP ${S.ep}` : ''}, DP ${S.dp}) and ${engines} inference engine${engines > 1 ? 's' : ''} of TP × PP = ${engine} GPUs, each holding one copy of the weights. Moving the training state out and back costs ${fmtBytes(2 * state)} per GPU per step over the CPU–GPU link; copying the updated weights into the inference engine moves ${fmtBytes(wInf)} per GPU.</p>
      <p class="caption">Colocated RL: the inference engine and the trainer take turns on the same GPUs. During generation the KV cache of all answers needs room, so gradients and optimizer state go to host memory and come back for training. In a disaggregated setup the two run on separate GPUs and only the weight copy crosses between them. Activations are approximate.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    const recipe = m.id === 'qwen3.5-397b' ? ' The defaults match the NeMo-RL GRPO recipe for this model: 32 prompts × 8 answers, prompts up to 2,048 tokens and prompt plus answer up to 8,192.' : '';
    return `
      <p class="lead">One RL step alternates between inference and training. An inference engine generates G answers for each prompt; a checker scores them; forward passes compute the token probabilities under the generating model and a frozen reference model; the trainer runs forward and backward passes over the answers and updates the weights; the new weights are copied into the inference engine for the next step.${recipe}</p>
      <h2>The phases of one step</h2>${figPhases(m, W)}
      <h2>GPU memory in each phase</h2>${figMemory(m, ctx, W)}
      <p class="sources">Sources: <a href="https://github.com/NVIDIA-NeMo/RL" target="_blank" rel="noreferrer">NeMo-RL</a> · <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">GRPO</a> · <a href="https://arxiv.org/abs/2409.19256" target="_blank" rel="noreferrer">HybridFlow (verl)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelectorAll('[data-rl]').forEach((el) => el.addEventListener('change', () => { W8[el.dataset.rl] = Number(el.value); rerender(); }));
  }

  (window.CHAPTERS = window.CHAPTERS || {})['rl-gpus'] = { render, mount };
})();
