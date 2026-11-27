// Chapter: Agents.
// Figure: tokens processed over the turns of an agent loop, with and without prefix caching, and the
// cache the conversation occupies for the selected model.
(function () {
  let turns = 20, perTurn = 1500, sys = 3000;

  function fig(m, W) {
    const ctxAt = (t) => sys + t * perTurn, noCache = [], withCache = [];
    let a = 0, b = 0;
    for (let t = 1; t <= turns; t++) { a += ctxAt(t); b += perTurn + (t === 1 ? sys : 0); noCache.push(a); withCache.push(b); }
    const H = 200, padL = 70, padB = 26, max = noCache[turns - 1], sx = (t) => padL + ((t - 1) / Math.max(1, turns - 1)) * (W - padL - 10), sy = (v) => 8 + (1 - v / max) * (H - padB - 8);
    let s = `<line x1="${padL}" x2="${W - 10}" y1="${H - padB}" y2="${H - padB}" class="grid"/><text x="${padL - 6}" y="${sy(max) + 4}" text-anchor="end" class="axis">${fmtParams(max)}</text><text x="${padL - 6}" y="${H - padB + 4}" text-anchor="end" class="axis">0</text>`;
    s += `<polyline points="${noCache.map((v, i) => `${sx(i + 1)},${sy(v)}`).join(' ')}" class="line-ink"/><polyline points="${withCache.map((v, i) => `${sx(i + 1)},${sy(v)}`).join(' ')}" class="line-base"/>`;
    s += `<text x="${sx(turns) - 4}" y="${sy(noCache[turns - 1]) + 14}" text-anchor="end" class="val">without prefix caching</text><text x="${sx(turns) - 4}" y="${sy(withCache[turns - 1]) - 6}" text-anchor="end" class="val">with prefix caching</text>`;
    s += `<text x="${padL}" y="${H - 6}" class="axis">turn 1</text><text x="${W - 10}" y="${H - 6}" text-anchor="end" class="axis">turn ${turns}</text>`;
    const final = ctxAt(turns), P = params(m);
    return `${figure(s, W, H, 'prefill tokens over turns')}
      <p class="status">After ${turns} turns the context is ${fmtN(final)} tokens: ${fmtBytes(cacheBytes(m, final).total)} of KV cache for ${esc(m.name)}. Prefilled in total: ${fmtParams(noCache[turns - 1])} tokens without prefix caching (${fmtFlops(2 * P.matmul * noCache[turns - 1])} for the weights alone), ${fmtParams(withCache[turns - 1])} with it.</p>
      <p class="caption">Cumulative prompt tokens processed by prefill. Without caching, every turn processes the whole conversation again, so the total grows with the square of the number of turns; with prefix caching, only each turn's new tokens, plus those of the last partly filled block, are processed, as long as the cached blocks stay in memory. Token counts per turn are illustrative.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    return `
      <p class="lead">An agent is a loop around the model: the model's reply contains a tool call, the software runs it (a search, code, a file edit), appends the result to the context, and calls the model again, until the reply contains no tool call. Each turn adds tokens, so the context and its KV cache grow with every step, and each call has to process the context before generating. Engines reuse the cache of the unchanged prefix between calls.</p>
      <p class="controls"><label class="nw">turns <select id="ag-t">${[5, 20, 50].map((v) => `<option${v === turns ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">tokens added per turn <select id="ag-p">${[500, 1500, 5000].map((v) => `<option value="${v}"${v === perTurn ? ' selected' : ''}>${fmtN(v)}</option>`).join('')}</select></label> <label class="nw">system prompt and tool definitions <select id="ag-s">${[1000, 3000, 10000].map((v) => `<option value="${v}"${v === sys ? ' selected' : ''}>${fmtN(v)}</option>`).join('')}</select></label></p>
      <h2>Tokens processed over the turns</h2>${fig(m, W)}
      <ul class="formulas">
        <li>Context limit: when the conversation approaches the model's maximum context (${fmtN(m.context)} for ${esc(m.name)}), agents summarize or drop older turns.</li>
        <li>Cache reuse needs the token IDs of the prefix to stay identical: a change at an early position (for example a timestamp in the system prompt) changes the hash of the block that contains it and of every later block, so prefill restarts from that block.</li>
      </ul>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2210.03629" target="_blank" rel="noreferrer">ReAct</a> · <a href="https://arxiv.org/abs/2312.07104" target="_blank" rel="noreferrer">SGLang (prefix reuse)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    const bind = (id, set) => root.querySelector(id).addEventListener('change', (e) => { set(Number(e.target.value)); rerender(); });
    bind('#ag-t', (v) => { turns = v; }); bind('#ag-p', (v) => { perTurn = v; }); bind('#ag-s', (v) => { sys = v; });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).agents = { render, mount };
})();
