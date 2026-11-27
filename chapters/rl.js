// Chapter: RL with checkable answers (GRPO).
// Figures: (1) one prompt, G sampled answers, their rewards and group-normalized advantages;
// (2) which models were post-trained with which method.
(function () {
  let G = 8, seed = 3;
  const PROMPT = 'A train travels 120 km in 1.5 hours. What is its average speed in km/h?';

  function sample() {
    const rnd = mulberry32(seed * 7919 + G);
    return range(G).map(() => {
      const correct = rnd() < 0.45, len = Math.round(200 + rnd() * 900);
      return { correct, len, answer: correct ? '80' : ['60', '90', '180', '75'][Math.floor(rnd() * 4)] };
    });
  }

  function fig(W) {
    const a = sample(), r = a.map((x) => (x.correct ? 1 : 0)), mean = r.reduce((s, x) => s + x, 0) / G;
    const sd = Math.sqrt(r.reduce((s, x) => s + (x - mean) ** 2, 0) / G), adv = r.map((x) => (sd > 0 ? (x - mean) / sd : 0));
    const rowH = 24, lw = 70, maxLen = 1100, bw = Math.min(260, W - lw - 260);
    let s = `<text x="0" y="12" class="axis">answer</text><text x="${lw}" y="12" class="axis">length (tokens)</text><text x="${lw + bw + 20}" y="12" class="axis">final result</text><text x="${lw + bw + 120}" y="12" class="axis">reward</text><text x="${lw + bw + 180}" y="12" class="axis">advantage</text>`;
    a.forEach((x, i) => {
      const y = 22 + i * rowH, A = adv[i], cx = lw + bw + 230;
      s += `<text x="0" y="${y + 13}" class="name">${i + 1}</text><rect x="${lw}" y="${y + 3}" width="${(x.len / maxLen) * bw}" height="12" class="seg-out"/>`
        + `<text x="${lw + bw + 20}" y="${y + 13}" class="tok">${x.answer} km/h</text><text x="${lw + bw + 120}" y="${y + 13}" class="name">${r[i]}</text>`
        + `<line x1="${cx}" x2="${cx + A * 22}" y1="${y + 9}" y2="${y + 9}" class="${A >= 0 ? 'adv-pos' : 'adv-neg'}"/><text x="${cx + (A >= 0 ? A * 22 + 6 : A * 22 - 6)}" y="${y + 13}"${A >= 0 ? '' : ' text-anchor="end"'} class="val">${A >= 0 ? '+' : ''}${A.toFixed(2)}</text>`;
    });
    const h = 22 + G * rowH;
    return `<p class="controls"><label class="nw">answers per prompt G <select id="rl-g">${[4, 8, 16].map((v) => `<option${v === G ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <a href="#" id="rl-resample">sample again</a></p>
      <p class="status">Prompt: <code>${esc(PROMPT)}</code> Checker: the final number equals 80.</p>
      ${figure(s, Math.max(W, lw + bw + 300), h, 'group of answers')}
      <p class="status">Mean reward ${mean.toFixed(2)}, standard deviation ${sd.toFixed(2)}. ${sd === 0 ? 'All answers scored the same, so every advantage is 0: the prompt contributes no policy-gradient term, and only the KL penalty (when β > 0) acts on its tokens. Dynamic sampling (DAPO, NeMo-RL) drops such groups before the update.' : 'The policy-gradient term raises the log-probability of every token of an answer with positive advantage and lowers it for every token of an answer with negative advantage; tokens whose probability ratio is already beyond the clip bound in that direction receive no gradient.'}</p>
      <p class="caption">Answers, lengths and correctness are illustrative; the page does not run the model. GRPO needs no value network: the group's mean reward is the baseline each answer is compared with.</p>`;
  }

  function figModels(m) {
    const rows = window.MODELS.map((x) => `<tr class="${x.id === m.id ? 'on' : ''}"><td>${esc(x.name)}</td><td class="f">${esc(x.postTraining?.text || '—')}${x.postTraining?.url ? ` <a href="${esc(x.postTraining.url)}" target="_blank" rel="noreferrer">source</a>` : ''}</td></tr>`).join('');
    return `<table class="data"><thead><tr><th>model</th><th>published post-training</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    return `
      <p class="lead">Supervised fine-tuning trains the model to reproduce example answers. Reinforcement learning (RL) trains it on answers it generated itself, according to how good they were. For problems whose result a program can check, such as math with a known answer or code with tests, the reward is computed automatically. GRPO (group relative policy optimization) generates G answers per prompt, scores them, and makes the above-average answers more probable and the below-average ones less probable.</p>
      <h2>One prompt, a group of answers</h2>${fig(W)}
      <h2>What training on rewards produced</h2>
      <p>DeepSeek-R1-Zero applied GRPO directly to the DeepSeek-V3 base model, with rewards only for a correct final answer and the required format. Over training, its pass@1 accuracy on AIME 2024 rose from 15.6% to 77.9% (DeepSeek-R1 paper v2; v1 reported 71.0% at an earlier training step), and its answers grew from hundreds to thousands of tokens: the model learned to write longer intermediate reasoning, including checking and revising its own steps, because that raised the reward. DeepSeek-R1 first fine-tunes on a few thousand long chain-of-thought examples in a readable format (cold start), runs reasoning RL, fine-tunes on about 800,000 rejection-sampled reasoning and non-reasoning samples, then runs a final RL stage over all prompt types.</p>
      <h2>Post-training of each model</h2>${figModels(m)}
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Advantage of answer i in a group: A<sub>i</sub> = (r<sub>i</sub> − mean(r)) / std(r), applied to every token of the answer.</li>
        <li>GRPO objective per token: min(ρ<sub>t</sub>·A<sub>i</sub>, clip(ρ<sub>t</sub>, 1 − ε, 1 + ε)·A<sub>i</sub>) − β·KL(π<sub>θ</sub> ‖ π<sub>ref</sub>), with ρ<sub>t</sub> = π<sub>θ</sub>(token) / π<sub>old</sub>(token), averaged over tokens and answers. The clip limits how far one update moves; the KL term keeps the model near a reference model.</li>
        <li>RL with verifiable rewards needs the answer generated in full before the reward exists, so each step contains a generation phase (One RL step on GPUs).</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath (introduces GRPO)</a> · <a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noreferrer">DeepSeek-R1</a> · <a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noreferrer">PPO</a> · <a href="https://arxiv.org/abs/2503.14476" target="_blank" rel="noreferrer">DAPO (dynamic sampling)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#rl-g').addEventListener('change', (e) => { G = Number(e.target.value); rerender(); });
    root.querySelector('#rl-resample').addEventListener('click', (e) => { e.preventDefault(); seed++; rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).rl = { render, mount };
})();
