// Chapter: Sampling from the base model.
// Figure: one next-token distribution (illustrative logits) with temperature and top-p applied,
// and draws from it. Per model: the output layer's size and FLOPs per generated token.
(function () {
  const CONTEXT = 'The cat sat on the';
  const CANDIDATES = [[' mat', 3.1], [' floor', 2.3], [' couch', 1.6], [' bed', 1.4], [' sofa', 1.3], [' windows', 0.9], [' roof', 0.8], [' edge', 0.6], [' table', 0.5], [' keyboard', 0.3], [' moon', -0.6], [' idea', -1.4]];
  const REST = 1.2;   // log of the summed probability mass of all other tokens, before temperature (illustrative)
  let temp = 1.0, topP = 0.9, draws = [], seed = 1;

  function dist() {
    const z = CANDIDATES.map(([, l]) => l / temp), rest = REST / temp + Math.log(1);   // the remaining vocabulary as one entry
    const all = [...z, rest], mx = Math.max(...all), e = all.map((v) => Math.exp(v - mx)), sum = e.reduce((a, b) => a + b, 0);
    const p = e.map((v) => v / sum);
    const order = CANDIDATES.map((_, i) => i).sort((a, b) => p[b] - p[a]);
    let cum = 0;
    const keep = new Set();
    for (const i of order) { if (cum >= topP) break; keep.add(i); cum += p[i]; }
    const kept = order.filter((i) => keep.has(i)), z2 = kept.reduce((s, i) => s + p[i], 0);
    return { p, keep, z2, restP: p[p.length - 1] };
  }

  function fig(W) {
    const d = dist(), lw = 110, bw = W - lw - 120, rowH = 22;
    let s = '';
    CANDIDATES.forEach(([t], i) => {
      const y = i * rowH, on = d.keep.has(i), pn = on ? d.p[i] / d.z2 : 0;
      s += `<text x="${lw - 8}" y="${y + 14}" text-anchor="end" class="tok">${showTok(t)}</text>`
        + `<rect x="${lw}" y="${y + 3}" width="${Math.max(0.5, d.p[i] * bw)}" height="14" class="${on ? 'pkeep' : 'pdrop'}"/>`
        + `<text x="${lw + bw + 8}" y="${y + 14}" class="val">${d.p[i].toFixed(3)}${on ? ` → ${pn.toFixed(3)}` : ' → 0'}</text>`;
    });
    const y = CANDIDATES.length * rowH;
    s += `<text x="${lw - 8}" y="${y + 14}" text-anchor="end" class="axis">all others</text><rect x="${lw}" y="${y + 3}" width="${d.restP * bw}" height="14" class="pdrop"/><text x="${lw + bw + 8}" y="${y + 14}" class="val">${d.restP.toFixed(3)} → 0</text>`;
    return figure(s, W, y + 24, 'next-token distribution');
  }

  function draw() {
    const d = dist(), rnd = mulberry32(seed++);
    let u = rnd() * d.z2;
    for (const i of [...d.keep]) { u -= d.p[i]; if (u <= 0) return CANDIDATES[i][0]; }
    return CANDIDATES[[...d.keep][0]][0];
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, H = m.hidden, V = m.vocab, P = params(m);
    return `
      <p class="lead">Generation repeats one step: run the network on the context, take the last position's probabilities over the vocabulary, choose one token, append it, and run again. Choosing the most probable token every time (greedy decoding) gives one fixed continuation. Sampling draws the token at random according to the probabilities, so the same context gives different continuations. Temperature and top-p change the distribution before the draw. A base model, trained only on documents, continues the text as a document would continue.</p>
      <p><b>${esc(m.name)}</b>: each step ends with the output layer, ${fmtN(H)} × ${fmtN(V)} = ${fmtParams(H * V)} weights, ${fmtFlops(2 * H * V)} per token (${fmtPct(H * V, P.matmul)} of the weights multiplied per token), and a softmax over ${fmtN(V)} entries.</p>
      <h2>One distribution, reshaped</h2>
      <p class="controls"><label class="nw">temperature <input id="sp-t" type="range" min="0.05" max="2" step="0.05" value="${temp}"></label> <span class="mono" id="sp-tv">${temp.toFixed(2)}</span> &nbsp; <label class="nw">top-p <input id="sp-p" type="range" min="0.05" max="1" step="0.05" value="${topP}"></label> <span class="mono" id="sp-pv">${topP.toFixed(2)}</span></p>
      <p class="status">Context: <code>${esc(CONTEXT)}</code> → next token?</p>
      <div id="sp-fig">${fig(W)}</div>
      <p class="controls"><a href="#" id="sp-draw">draw 10 tokens</a> <span class="mono" id="sp-out">${draws.map(showTok).join(' ')}</span></p>
      <p class="caption">Probabilities after temperature (bar) and after top-p (right of the arrow): the most probable tokens are kept until their total reaches p, the rest get probability 0, and the kept ones are rescaled to sum to 1. Logits are illustrative; the page does not run the model. Temperature below 1 sharpens the distribution toward greedy decoding; above 1 it flattens it.</p>
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Temperature τ: p<sub>i</sub> = exp(z<sub>i</sub>/τ) / Σ<sub>j</sub> exp(z<sub>j</sub>/τ). τ → 0 gives greedy decoding.</li>
        <li>Top-p: sort by p, keep the smallest set with Σ p ≥ p<sub>top</sub>, renormalize. Top-k keeps the k largest instead.</li>
        <li>One generated token needs one forward pass for one new position: ${fmtFlops(2 * P.matmul)} for the weights, plus attention over the cached positions (Prefill and decode).</li>
      </ul></details>`;
  }

  function mount(root, ctx) {
    const update = () => { root.querySelector('#sp-fig').innerHTML = fig(ctx.width); root.querySelector('#sp-tv').textContent = temp.toFixed(2); root.querySelector('#sp-pv').textContent = topP.toFixed(2); };
    root.querySelector('#sp-t').addEventListener('input', (e) => { temp = Number(e.target.value); update(); });
    root.querySelector('#sp-p').addEventListener('input', (e) => { topP = Number(e.target.value); update(); });
    root.querySelector('#sp-draw').addEventListener('click', (e) => { e.preventDefault(); draws = range(10).map(draw); root.querySelector('#sp-out').innerHTML = draws.map(showTok).join(' '); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).sampling = { render, mount };
})();
