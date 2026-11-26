// Chapter: Inference engines.
// Figures: (1) the engine settings that divide GPU memory, applied to the selected model; (2) the
// mechanisms engines use; (3) measured next to computed.
(function () {
  let wprec = 'release', util = 0.9, maxLen = 32768, kvDtype = 'bf16';

  function figSettings(m, ctx, W) {
    const G = ctx.gpus, mem = ctx.mem, wPer = weightBytes(m, wprec === 'release' ? {} : { experts: wprec, linear: wprec }).total / G, act = 0.05 * mem, pool = util * mem - wPer - act;
    const tp = Math.min(G, 8), perTok = kvBytesPerTokenPerGPU(m, tp, kvDtype) / Math.max(1, G / tp);
    const k = (W - 4) / mem;
    let s = `<rect x="0" y="18" width="${wPer * k}" height="24" class="w sep"><title>weights</title></rect><rect x="${wPer * k}" y="18" width="${act * k}" height="24" class="a sep"><title>activations, CUDA graphs</title></rect>`;
    if (pool > 0) s += `<rect x="${(wPer + act) * k}" y="18" width="${pool * k}" height="24" class="kvbar sep"><title>KV pool</title></rect>`;
    s += `<line x1="${util * mem * k}" x2="${util * mem * k}" y1="10" y2="50" class="memline"/><text x="${Math.min(util * mem * k, W - 70)}" y="8" text-anchor="middle" class="axis">--gpu-memory-utilization ${util}</text>`;
    const tokens = pool > 0 ? Math.floor(pool / perTok) : 0;
    return `<p class="controls"><label class="nw">weights <select id="en-w">${[['release', 'published release'], ['bf16', 'BF16'], ['fp8', 'FP8']].map(([v, n]) => `<option value="${v}"${v === wprec ? ' selected' : ''}>${n}</option>`).join('')}</select></label> <label class="nw">memory fraction <select id="en-u">${[0.8, 0.85, 0.9, 0.95].map((v) => `<option${v === util ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">max context <select id="en-l">${(([8192, 32768, 131072, 262144].filter((v) => v <= m.context)).length ? [8192, 32768, 131072, 262144].filter((v) => v <= m.context) : [m.context]).map((v) => `<option value="${v}"${v === maxLen ? ' selected' : ''}>${fmtN(v)}</option>`).join('')}</select></label> <label class="nw">KV dtype <select id="en-k"><option value="bf16"${kvDtype === 'bf16' ? ' selected' : ''}>auto (BF16)</option><option value="fp8"${kvDtype === 'fp8' ? ' selected' : ''}>fp8</option></select></label></p>
      ${figure(s, W, 52, 'engine memory split')}
      <p class="status">${pool > 0 ? `The KV pool holds ${fmtN(tokens)} tokens per instance, ${(tokens / maxLen).toFixed(1)} sequences at the maximum context of ${fmtN(maxLen)}.` : 'The weights alone exceed the memory fraction: add GPUs (tensor or pipeline parallelism) or use a smaller precision.'}</p>
      <pre class="doc">vllm serve ${esc(m.source.split('huggingface.co/')[1].split('/blob')[0])} \\
    --tensor-parallel-size ${Math.min(G, 8)}${G > 8 ? ` --pipeline-parallel-size ${Math.ceil(G / 8)}` : ''}${isMoE(m) ? ' --enable-expert-parallel' : ''} \\
    --max-model-len ${maxLen} --gpu-memory-utilization ${util}${kvDtype === 'fp8' ? ' --kv-cache-dtype fp8' : ''}</pre>
      <p class="caption">How the settings divide each GPU (vLLM names; other engines have equivalents): the memory fraction caps what the engine uses; weights load first; a profiling run measures activation memory; the rest becomes the KV pool, in blocks of 16 positions. --max-model-len only has to fit one sequence; the pool is shared by all running sequences. The command is illustrative; check flag names against your engine version.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width;
    if (maxLen > m.context) maxLen = m.context;
    return `
      <p class="lead">An inference engine runs a model for many users at once. It loads the weights across the GPUs, keeps a pool of GPU memory for KV caches, and at every step chooses which sequences to run, combining prefill of new prompts with decode of running ones. vLLM, SGLang and TensorRT-LLM serve on data-center GPUs; llama.cpp, Ollama and MLX run quantized models on laptops and desktops.</p>
      <h2>Settings that divide GPU memory</h2>${figSettings(m, ctx, W)}
      <h2>Mechanisms</h2>
      <ul class="formulas">
        <li><b>Paged KV cache:</b> the cache is allocated in fixed-size blocks through a block table per sequence, so memory is not reserved for the maximum length of every request.</li>
        <li><b>Continuous batching:</b> finished sequences leave and new ones join at every step, so the decode batch stays full.</li>
        <li><b>Prefix caching:</b> blocks of a prompt prefix that an earlier request already computed (a system prompt, earlier conversation turns) are reused instead of prefilled again.</li>
        <li><b>Chunked prefill:</b> long prompts are split into chunks that share steps with decode, so running sequences keep producing tokens.</li>
        <li><b>Speculative decoding:</b> a draft (a small model, or an MTP module as in DeepSeek-V3 and Qwen3.5) proposes several tokens; one forward pass of the full model checks them all.</li>
        <li><b>CUDA graphs:</b> the kernel launches of a decode step are recorded once and replayed, which removes CPU launch overhead for small batches.</li>
        <li><b>Parallel serving:</b> tensor parallelism inside a node, pipeline parallelism across nodes, expert parallelism for MoE layers, data-parallel attention; separate GPU pools for prefill and decode (disaggregation, as in DeepSeek-V3's deployment).</li>
      </ul>
      ${measuredTable('engines', m)}
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">vLLM / PagedAttention</a> · <a href="https://arxiv.org/abs/2312.07104" target="_blank" rel="noreferrer">SGLang (RadixAttention prefix caching)</a> · <a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">Orca (continuous batching)</a> · <a href="https://arxiv.org/abs/2401.09670" target="_blank" rel="noreferrer">DistServe (disaggregation)</a> · <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer">llama.cpp</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#en-w').addEventListener('change', (e) => { wprec = e.target.value; rerender(); });
    root.querySelector('#en-u').addEventListener('change', (e) => { util = Number(e.target.value); rerender(); });
    root.querySelector('#en-l').addEventListener('change', (e) => { maxLen = Number(e.target.value); rerender(); });
    root.querySelector('#en-k').addEventListener('change', (e) => { kvDtype = e.target.value; rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).engines = { render, mount };
})();
