// Chapter: Parameters and training tokens.
// Figures: (1) models placed by parameters N and training tokens D, with the 20-tokens-per-parameter
// line and lines of equal compute 6·N·D; (2) a compute budget split into N and D at 20 tokens per
// parameter.
(function () {
  let logC = 25.58;   // log10 of the compute budget in FLOPs (3.8 × 10^25)

  function fig(m, W) {
    const H = 300, padL = 54, padB = 30, padR = 16, sx = logScale(1e9, 2e12, padL, W - padR);
    const sy = (d) => 8 + (1 - (Math.log10(d) - 10) / (14 - 10)) * (H - padB - 8);
    let s = '';
    [1e9, 1e10, 1e11, 1e12].forEach((n) => { s += `<line x1="${sx(n)}" x2="${sx(n)}" y1="8" y2="${H - padB}" class="grid"/><text x="${sx(n)}" y="${H - padB + 16}" text-anchor="middle" class="axis">${fmtParams(n)}</text>`; });
    [1e10, 1e11, 1e12, 1e13, 1e14].forEach((d) => { s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(d)}" y2="${sy(d)}" class="grid"/><text x="${padL - 6}" y="${sy(d) + 4}" text-anchor="end" class="axis">${fmtParams(d)}</text>`; });
    s += `<text x="${W - padR}" y="${H - 4}" text-anchor="end" class="axis">parameters N →</text><text x="${padL + 4}" y="18" class="axis">training tokens D</text>`;
    // 20 tokens per parameter
    s += `<line x1="${sx(1e9)}" y1="${sy(2e10)}" x2="${sx(2e12)}" y2="${sy(4e13)}" class="chin"/><text x="${sx(2e10)}" y="${sy(4e11) - 6}" class="axis">D = 20·N</text>`;
    // equal compute: D = C / (6N)
    [1e22, 1e23, 1e24, 1e25].forEach((C) => {
      const n0 = Math.max(1e9, C / 6 / 1e14), n1 = Math.min(2e12, C / 6 / 1e10);
      s += `<line x1="${sx(n0)}" y1="${sy(C / 6 / n0)}" x2="${sx(n1)}" y2="${sy(C / 6 / n1)}" class="iso"/><text x="${sx(n1) - 2}" y="${sy(C / 6 / n1) - 4}" text-anchor="end" class="axis">${fmtExpText(C)} FLOPs</text>`;
    });
    const pts = window.MODELS.filter((x) => x.training?.tokens).sort((a, b) => params(a).matmul - params(b).matmul);
    pts.forEach((x, k) => {
      const P = params(x), D = x.training.tokens, on = x.id === m.id, px = sx(P.matmul), py = sy(D);
      if (isMoE(x)) s += `<line x1="${px}" x2="${sx(P.total)}" y1="${py}" y2="${py}" class="span"/><circle cx="${sx(P.total)}" cy="${py}" r="5" class="tot"/>`;
      const right = px < W * 0.6, ly = k % 2 ? py + 17 : py - 9;
      s += `<g class="row${on ? ' on' : ''}" data-model="${x.id}"><circle cx="${px}" cy="${py}" r="${on ? 6 : 4.5}" class="act"><title>${esc(x.name)}: N = ${fmtParams(P.matmul)}${isMoE(x) ? ` active (${fmtParams(P.total)} total)` : ''}, D = ${fmtParams(D)} tokens</title></circle>`
        + `<text x="${right ? px + 8 : px - 8}" y="${ly}"${right ? '' : ' text-anchor="end"'} class="${on ? 'name' : 'val'}">${esc(x.short)}: ${fmtN(D / P.matmul)} tokens/param</text></g>`;
    });
    // compute-optimal point for the chosen budget
    const C = 10 ** logC, Nopt = Math.sqrt(C / 120), Dopt = 20 * Nopt;
    if (Nopt >= 1e9 && Nopt <= 2e12) s += `<circle cx="${sx(Nopt)}" cy="${sy(Dopt)}" r="5" class="opt"/><text x="${sx(Nopt) + 8}" y="${sy(Dopt) + 18}" class="axis">budget at 20 tokens/param</text>`;
    return figure(s, W, H, 'parameters and training tokens');
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, P = params(m), C = 10 ** logC, Nopt = Math.sqrt(C / 120), Dopt = 20 * Nopt, tr = m.training;
    const own = tr?.tokens
      ? `<b>${esc(m.name)}</b>: ${fmtParams(P.matmul)} weights used per token${isMoE(m) ? ` (${fmtParams(P.total)} stored)` : ''}, trained on ${esc(tr.text)}: ${fmtN(tr.tokens / P.matmul)} tokens per weight used, compute 6·N·D = ${fmtExp(6 * P.matmul * tr.tokens)} FLOPs.`
      : `The authors of <b>${esc(m.name)}</b> have not published its number of training tokens, so it is not in the figure.`;
    return `
      <p class="lead">Training compute is about C = 6·N·D FLOPs for N weights used per token and D training tokens. A fixed budget can buy a larger model trained on fewer tokens, or a smaller model trained on more. Hoffmann et al. (2022) trained over 400 models of 70M to 16B parameters on 5B to 500B tokens and found that, for a fixed budget, the loss-minimizing N and D grow in equal proportion; their Table 3 gives about 20 tokens per parameter (67B parameters on 1.5T tokens), while their parametric fit gives more tokens per parameter. Models that serve many requests are trained well beyond that ratio, because a smaller model costs less for every generated token, and extra tokens keep lowering its loss.</p>
      <p>${own}</p>
      <h2>Parameters and tokens of the models</h2>${fig(m, W)}
      <p class="caption">Dots: models with published training tokens, at their weights used per token (for MoE models, a ring marks the total stored). Diagonal: 20 tokens per parameter. Gray lines: equal compute C = 6·N·D. The model names give D/N. Click a dot to select that model.</p>
      <h2>Splitting a budget</h2>
      <p class="controls"><label>Compute budget <input id="sc-c" type="range" min="21" max="26.5" step="0.01" value="${logC}"></label> <span id="sc-val">${fmtExp(C)} FLOPs</span></p>
      <p class="status">At 20 tokens per parameter, C = 6·N·(20·N) gives N = √(C / 120) = <b>${fmtParams(Nopt)}</b> parameters and D = <b>${fmtParams(Dopt)}</b> tokens.${Math.abs(logC - 25.58) < 0.02 ? ' For 3.8 × 10<sup>25</sup> FLOPs, the Llama 3 paper\'s own scaling-law fit predicted 402B parameters and 16.55T tokens; Llama 3.1 405B was trained with 15.6T.' : ''}</p>
      <p class="caption">The 20 tokens per parameter ratio comes from runs at 6 × 10<sup>18</sup> to 3 × 10<sup>21</sup> FLOPs, extrapolated to Gopher's budget of 5.76 × 10<sup>23</sup> FLOPs, where Chinchilla (70B parameters, 1.4T tokens) was trained to test it. Later fits give other ratios: Llama 3's gives 16.55T tokens for 402B parameters (41 per parameter). The ratio ignores the cost of using the model after training.</p>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer">Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)</a> · <a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noreferrer">Llama 3 paper, section 3.2.1</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    const el = root.querySelector('#sc-c');
    el.addEventListener('change', () => { logC = Number(el.value); rerender(); });
    el.addEventListener('input', () => { logC = Number(el.value); root.querySelector('#sc-val').innerHTML = `${fmtExp(10 ** logC)} FLOPs`; const st = root.querySelector('.status'); if (st) { const C = 10 ** logC, N = Math.sqrt(C / 120); st.innerHTML = `At 20 tokens per parameter, C = 6·N·(20·N) gives N = √(C / 120) = <b>${fmtParams(N)}</b> parameters and D = <b>${fmtParams(20 * N)}</b> tokens.`; } });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).scaling = { render, mount };
})();
