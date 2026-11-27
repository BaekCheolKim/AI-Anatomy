// Measured values from the author's runs, shown next to the computed values in their chapters.
// Each states its conditions. This repository is public: no job IDs, node names, host names or paths.
// computed(m): the page's value for the same quantity, from calc.js (null when there is none).

window.MEASURED = [
  {
    chapter: 'kv-cache', model: 'qwen3-vl-235b', what: 'KV cache per token per GPU',
    value: '48,144 B (vLLM reported 22.22 GiB of KV cache for 495,568 tokens)',
    computed: (m) => `${fmtN(kvBytesPerTokenPerGPU(m, 4))} B (94 layers × 2 × 1 K/V head of 128 × 2 B)`,
    conditions: 'vLLM 0.32.0, FP8 weights, BF16 KV cache, TP 4 on 4 × GH200 96 GB, November 2026',
  },
  {
    chapter: 'precision', model: 'qwen3-vl-235b', what: 'FP8 checkpoint size',
    value: '238 GB on disk (222 GiB)',
    computed: (m) => `${fmtBytes(weightBytes(m, { experts: 'fp8', linear: 'fp8' }).total)}: FP8 linear layers and experts, BF16 embedding, output layer, norms, routers and vision encoder`,
    conditions: 'Qwen3-VL-235B-A22B-Instruct-FP8 as published',
  },
  {
    chapter: 'precision', model: 'qwen3-vl-235b', what: 'Weights per GPU after loading',
    value: '56.47 GiB',
    computed: (m) => { const W = weightBytes(m, { experts: 'fp8', linear: 'fp8' }); return `${((W.total - W.byClass.vision) / 4 / 2 ** 30 + W.byClass.vision / 2 ** 30).toFixed(2)} GiB: the language model split over 4 GPUs, the vision encoder copied to each (mm-encoder-tp-mode data)`; },
    conditions: 'vLLM 0.32.0, TP 4 on 4 × GH200 96 GB',
  },
  {
    chapter: 'engines', model: 'qwen3-vl-235b', what: 'Model loading',
    value: '167.8 s; each of the 4 processes read the whole 238 GB checkpoint from a Lustre file system',
    computed: (m) => `${fmtBytes(weightBytes(m, { experts: 'fp8', linear: 'fp8' }).total / 4)} per GPU if each process read only its own quarter`,
    conditions: 'vLLM 0.32.0, TP 4, one node, checkpoint on a shared parallel file system',
  },
  {
    chapter: 'engines', model: 'qwen3-vl-235b', what: 'GPU memory split by the engine',
    value: '56.47 GiB weights + 22.22 GiB KV cache within 0.90 × 95.6 GiB (97,871 MiB per GPU); the rest for activations, CUDA graphs and allocations outside PyTorch (for example NCCL buffers)',
    computed: null,
    conditions: 'vLLM 0.32.0, --gpu-memory-utilization 0.90, --max-model-len 131072',
  },
  {
    chapter: 'serving', model: 'qwen3-vl-235b', what: 'Output tokens per second, 64 concurrent text requests',
    value: '1,394.65 tokens/s (5.45 requests/s)',
    computed: null,
    conditions: 'vLLM 0.32.0, FP8, TP 4 on 4 × GH200 96 GB, one benchmark run, untuned',
  },
  {
    chapter: 'serving', model: 'qwen3-vl-235b', what: 'Output tokens per second, 32 concurrent image + text requests',
    value: '1,152.95 tokens/s (4.50 requests/s)',
    computed: null,
    conditions: 'same as above',
  },
  {
    chapter: 'parallel', model: null, what: 'All-reduce of 512 MiB over 2 nodes × 4 GPUs',
    value: '3.4 ms',
    computed: () => `${fmtBytes(collectiveBytes('all-reduce', 8, 4, 512 * 2 ** 20).inter)} sent between nodes per GPU, ${fmtBytes(collectiveBytes('all-reduce', 8, 4, 512 * 2 ** 20).intra)} inside the node; at 50 GB/s per GPU between nodes that is ≥ 2.7 ms`,
    conditions: 'NCCL with the Libfabric plugin, 4 × 400 Gb/s Slingshot NICs per node (one per GPU)',
  },
];

// Table of the measured values for one chapter (and optionally one model).
function measuredTable(chapter, m) {
  const rows = window.MEASURED.filter((x) => x.chapter === chapter);
  if (!rows.length) return '';
  const models = (id) => window.MODELS.find((x) => x.id === id);
  return `<h2>Measured next to computed</h2>
    <table class="data"><thead><tr><th>quantity</th><th>computed</th><th>measured</th><th>conditions</th></tr></thead><tbody>
    ${rows.map((r) => `<tr class="${r.model && m && r.model === m.id ? 'on' : ''}"><td>${esc(r.what)}${r.model ? `<br><span class="f">${esc(models(r.model).name)}</span>` : ''}</td><td class="f">${r.computed ? r.computed(r.model ? models(r.model) : null) : '—'}</td><td>${esc(r.value)}</td><td class="f">${esc(r.conditions)}</td></tr>`).join('')}
    </tbody></table>
    <p class="caption">Measured values come from runs on one system and depend on its hardware and software. The computed column comes from the model's config and the page's formulas; where it states a bandwidth, it holds only for that bandwidth.</p>`;
}
