// Chapter: Will it fit?
// Figure: for every model, the minimum number of GPUs of the reader's memory size for the weights at
// a chosen precision plus the KV cache of N sequences at a chosen length.
(function () {
  let prec = 'release', seqs = 16, len = 8192, kvDtype = 'bf16';
  const PREC = { release: 'published release', bf16: 'BF16', fp8: 'FP8', int4: '4-bit (≈ 4.5 bits)' };
  const bytesFor = (m) => (prec === 'release' ? weightBytes(m).total
    : prec === 'int4' ? (() => { const c = weightClasses(m); return (c.experts + c.linear) * 0.5625 + (c.embed + c.small + c.vision) * 2; })()
      : weightBytes(m, { experts: prec, linear: prec }).total);

  function render(ctx) {
    const m = ctx.m, mem = ctx.mem, util = 0.9;
    const rows = window.MODELS.map((x) => {
      const w = bytesFor(x), kv = seqs * cacheBytes(x, Math.min(len, x.context), kvDtype).total, need = w + kv + 0.05 * mem;
      const g = Math.ceil(need / (util * mem));
      return `<tr class="${x.id === m.id ? 'on' : ''}"><td>${esc(x.name)}</td><td class="r">${fmtBytes(w)}</td><td class="r">${fmtBytes(kv)}</td><td class="r"><b>${g}</b></td><td class="r">${g <= ctx.gpus ? 'fits' : `needs ${g - ctx.gpus} more`}</td></tr>`;
    }).join('');
    return `
      <p class="lead">To run a model, the GPUs together must hold its weights, the KV cache of every sequence being served, and working memory for activations. Weights are fixed for a given precision; the cache grows with the number of sequences and their length. This chapter adds them up for every model, against the GPU memory and GPU count set at the top.</p>
      <p class="controls"><label class="nw">weights <select id="ft-p">${Object.entries(PREC).map(([k, v]) => `<option value="${k}"${k === prec ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">sequences <select id="ft-n">${[1, 4, 16, 64, 256].map((v) => `<option${v === seqs ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">tokens each <select id="ft-l">${[2048, 8192, 32768, 131072].map((v) => `<option value="${v}"${v === len ? ' selected' : ''}>${fmtN(v)}</option>`).join('')}</select></label> <label class="nw">KV <select id="ft-k"><option value="bf16"${kvDtype === 'bf16' ? ' selected' : ''}>BF16</option><option value="fp8"${kvDtype === 'fp8' ? ' selected' : ''}>FP8</option></select></label></p>
      <table class="data"><thead><tr><th>model</th><th class="r">weights</th><th class="r">KV cache</th><th class="r">GPUs of ${fmtBytes(mem)}</th><th class="r">with ${ctx.gpus} GPU${ctx.gpus > 1 ? 's' : ''}</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="caption">Minimum GPUs = ⌈(weights + KV cache + 5% of one GPU for activations) / (0.9 × GPU memory)⌉, the 0.9 leaving room for the runtime. Lengths above a model's maximum context are capped at it. Real deployments need more: weights split unevenly over GPUs, parallel layouts replicate some tensors, and engines reserve memory for CUDA graphs and buffers.</p>
      <p>For running locally, GPU memory can be replaced by the memory available to the accelerator: unified memory on laptops with integrated GPUs, or CPU memory for layers or experts offloaded from the GPU, at the speed of the slower memory.</p>`;
  }

  function mount(root, ctx, rerender) {
    const bind = (id, set) => root.querySelector(id).addEventListener('change', (e) => { set(e.target.value); rerender(); });
    bind('#ft-p', (v) => { prec = v; }); bind('#ft-n', (v) => { seqs = Number(v); }); bind('#ft-l', (v) => { len = Number(v); }); bind('#ft-k', (v) => { kvDtype = v; });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).fit = { render, mount };
})();
