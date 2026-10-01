// Deep-dive topics. Each one gets a page at technique.html?id=<id> (template only for now).
// `uses(m)` returns true for models in models.js that the technique applies to; `null` means not model-specific.

window.TECHNIQUES = [
  // attention and model structure
  { id: 'flash-attention', title: 'FlashAttention', group: 'Attention', uses: null },
  { id: 'gqa', title: 'Grouped-query attention (GQA)', group: 'Attention', uses: (m) => m.attention.kind === 'GQA' },
  { id: 'mla', title: 'Multi-head latent attention (MLA)', group: 'Attention', uses: (m) => m.attention.kind === 'MLA' },
  { id: 'rope', title: 'Rotary position embedding (RoPE)', group: 'Attention', uses: (m) => m.position === 'rope' },
  { id: 'swiglu', title: 'Gated FFN (SwiGLU)', group: 'Feed-forward', uses: (m) => m.ffn.gated },
  { id: 'moe', title: 'Mixture of experts and routing', group: 'Feed-forward', uses: (m) => m.ffn.kind === 'moe' },

  // inference
  { id: 'kv-cache', title: 'KV cache', group: 'Inference', uses: null },
  { id: 'paged-kv-cache', title: 'Paged KV cache (PagedAttention)', group: 'Inference', uses: null },
  { id: 'continuous-batching', title: 'Continuous batching', group: 'Inference', uses: null },
  { id: 'prefill-decode', title: 'Prefill vs decode and disaggregated serving', group: 'Inference', uses: null },
  { id: 'roofline', title: 'Arithmetic intensity and the roofline model', group: 'Hardware', uses: null },

  // training
  { id: 'mixed-precision', title: 'Mixed precision (BF16 / FP8)', group: 'Training', uses: null },
  { id: 'activation-checkpointing', title: 'Activation checkpointing (recompute)', group: 'Training', uses: null },

  // parallelism
  { id: 'data-parallel', title: 'Data parallelism (DP)', group: 'Parallelism', uses: null },
  { id: 'zero', title: 'ZeRO stages 1 / 2 / 3', group: 'Parallelism', uses: null },
  { id: 'fsdp', title: 'Fully sharded data parallel (FSDP)', group: 'Parallelism', uses: null },
  { id: 'tensor-parallel', title: 'Tensor parallelism (TP)', group: 'Parallelism', uses: null },
  { id: 'sequence-parallel', title: 'Sequence parallelism (SP)', group: 'Parallelism', uses: null },
  { id: 'context-parallel', title: 'Context parallelism (CP) and ring attention', group: 'Parallelism', uses: null },
  { id: 'pipeline-parallel', title: 'Pipeline parallelism (PP) and schedules', group: 'Parallelism', uses: null },
  { id: 'expert-parallel', title: 'Expert parallelism (EP)', group: 'Parallelism', uses: (m) => m.ffn.kind === 'moe' },
  { id: 'collectives', title: 'Collective operations (all-reduce, all-gather, reduce-scatter, all-to-all, P2P)', group: 'Hardware', uses: null },
];

// Section headings every deep-dive page uses.
window.TECHNIQUE_SECTIONS = [
  'Definition',
  'Operation and formulas',
  'Tensors and shapes',
  'Memory: residency and lifetime',
  'Data movement and collectives',
  'Compute and bandwidth cost',
  'Training vs inference',
  'Models in this page that use it',
  'References',
];
