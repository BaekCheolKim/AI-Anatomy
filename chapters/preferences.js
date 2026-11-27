// Chapter: Preference training (RLHF with PPO, and DPO).
// Figures: (1) one preference pair with reward-model scores and the probability the reward model
// assigns to the human's choice; (2) models held in memory by PPO, GRPO and DPO for the selected model.
(function () {
  let rc = 1.3, rr = -0.4;

  function figPair(W) {
    const p = 1 / (1 + Math.exp(-(rc - rr)));
    return `<table class="data"><thead><tr><th></th><th>answer to “Write a one-line summary of photosynthesis.”</th><th class="r">reward-model score</th></tr></thead><tbody>
      <tr><td>chosen</td><td class="f">Plants use light to turn carbon dioxide and water into sugar and oxygen.</td><td class="r"><input id="pf-c" type="number" step="0.1" value="${rc}" class="num-in"></td></tr>
      <tr><td>rejected</td><td class="f">Photosynthesis is a very important process that happens in many living things in nature.</td><td class="r"><input id="pf-r" type="number" step="0.1" value="${rr}" class="num-in"></td></tr></tbody></table>
      <p class="status">P(chosen preferred) = σ(r<sub>chosen</sub> − r<sub>rejected</sub>) = σ(${(rc - rr).toFixed(1)}) = <b>${p.toFixed(3)}</b>; this pair's loss −ln ${p.toFixed(3)} = ${(-Math.log(p)).toFixed(3)}.</p>
      <p class="caption">A person saw both answers and chose one. The reward model is a transformer language model initialized from a pre-trained or fine-tuned checkpoint, not necessarily of the policy's size (InstructGPT: a 6B reward model for a 175B policy), with the output layer replaced by a linear layer that outputs one number. It is trained to score the chosen answer higher. Scores are illustrative; change them to see the loss.</p>`;
  }

  function figMemory(m, ctx, W) {
    const P = params(m).total, tr = P * 18, frozen = P * 2;
    const setups = [
      ['PPO (RLHF)', [['g', 'policy, trained', tr], ['g-dark', 'value model, trained', tr], ['w', 'reference model, frozen', frozen], ['w', 'reward model, frozen', frozen]]],
      ['GRPO', [['g', 'policy, trained', tr], ['w', 'reference model, frozen', frozen]]],
      ['DPO', [['g', 'policy, trained', tr], ['w', 'reference model, frozen', frozen]]],
    ];
    const max = Math.max(...setups.map(([, s]) => s.reduce((t, x) => t + x[2], 0))), lw = W < 560 ? 0 : 110, bw = W - lw - (W < 560 ? 0 : 90), rowH = W < 560 ? 40 : 30;
    let s = '';
    setups.forEach(([name, segs], i) => {
      const y = i * rowH + (W < 560 ? 14 : 2); let x = lw;
      s += `<text x="0" y="${W < 560 ? y : y + 15}" class="name">${name}</text>`;
      segs.forEach(([k, label, v]) => { s += `<rect x="${x}" y="${W < 560 ? y + 4 : y + 2}" width="${(v / max) * bw}" height="18" class="${k} sep"><title>${label}: ${fmtBytes(v)}</title></rect>`; x += (v / max) * bw; });
      const tot = segs.reduce((t, z) => t + z[2], 0);
      s += `<text x="${W < 560 ? W : lw + bw + 8}" y="${W < 560 ? y : y + 15}"${W < 560 ? ' text-anchor="end"' : ''} class="val">${fmtBytes(tot)}</text>`;
    });
    return `${figure(s, W, setups.length * rowH + (W < 560 ? 14 : 4), 'models in memory')}
      <p class="caption">Models held during training if each is as large as ${esc(m.name)} (${fmtParams(P)} parameters): trained models with their full training state (18 B per parameter, vermillion), frozen models as BF16 weights (2 B per parameter, green). That is at least ${fmtN(Math.ceil((tr * 2 + frozen * 2) / ctx.mem))} GPUs of ${fmtBytes(ctx.mem)} for PPO and ${fmtN(Math.ceil((tr + frozen) / ctx.mem))} for GRPO or DPO, before activations and KV cache. Reward and value models are often smaller than the policy.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    return `
      <p class="lead">For answers that no program can check, such as a summary, a poem or a polite refusal, the training signal comes from people comparing answers. In RLHF (reinforcement learning from human feedback), a reward model is trained on these comparisons and the language model is then optimized with PPO to get high scores from it. DPO (direct preference optimization) trains the language model on the comparisons directly, with no reward model and no generation during training.</p>
      <h2>One comparison</h2>${figPair(W)}
      <h2>Models in memory</h2>${figMemory(m, ctx, W)}
      <h2>Limits</h2>
      <ul class="formulas">
        <li>The reward model is itself a network trained on finite data. Optimizing against it for many steps finds answers it scores highly that people would not prefer, so RLHF adds a per-token KL penalty toward the starting model (InstructGPT) and retrains the reward model on outputs of the current policy in each round (Llama 2).</li>
        <li>Comparing two answers is easier for people than writing the best answer, which is why preference data adds to demonstration data.</li>
      </ul>
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Reward model (Bradley–Terry): L = −ln σ(r(x, y<sub>w</sub>) − r(x, y<sub>l</sub>)) for chosen y<sub>w</sub> and rejected y<sub>l</sub>.</li>
        <li>RLHF objective: maximize E[r(x, y)] − β·KL(π<sub>θ</sub>(y|x) ‖ π<sub>ref</sub>(y|x)), optimized with PPO, which also trains a value model that predicts, at each token position, the expected return (the rewards from that position to the end of the answer, including per-token KL penalties); generalized advantage estimation (GAE) computes the advantages from it.</li>
        <li>DPO: L = −ln σ(β·[ln π<sub>θ</sub>(y<sub>w</sub>|x)/π<sub>ref</sub>(y<sub>w</sub>|x) − ln π<sub>θ</sub>(y<sub>l</sub>|x)/π<sub>ref</sub>(y<sub>l</sub>|x)]): the same objective solved in closed form for the policy, so only the log-probabilities of the two answers are needed.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noreferrer">InstructGPT</a> · <a href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noreferrer">DPO (Rafailov et al., 2023)</a> · <a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">Llama 2 paper (RLHF with PPO)</a> · <a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noreferrer">Llama 3 paper (DPO)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#pf-c').addEventListener('change', (e) => { rc = Number(e.target.value) || 0; rerender(); });
    root.querySelector('#pf-r').addEventListener('change', (e) => { rr = Number(e.target.value) || 0; rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).preferences = { render, mount };
})();
