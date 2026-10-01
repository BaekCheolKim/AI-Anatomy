// Model families and models. Every config value comes from the model's published config.json
// (or paper, where noted in `sources`). app.js and calc.js build the page from these fields only.
//
// attention.kind: 'MHA' | 'GQA' | 'MLA'      attention.bias: 'none' | 'qkv' | 'all'
// ffn.kind:       'dense' | 'moe'             ffn.act: 'silu' | 'gelu'   ffn.gated: SwiGLU-style gate
// position:       'learned' | 'rope'          norm: 'LayerNorm' | 'RMSNorm'
// why:            the technique or scale this model is included to show
// presets:        a parallel layout + workload per phase; `note` says which values are published
// trainPrecision: overrides of the default mixed-precision layout (see calc.js TRAIN_BYTES)

window.FAMILIES = [
  { id: 'llm', title: 'Decoder-only LLM', status: 'ready' },
  {
    id: 'vit', title: 'Vision Transformer (ViT)', status: 'template',
    outline: [
      ['input', 'Image (H × W × 3)'], ['input', 'Patchify (P × P patches)'], ['input', 'Linear patch embedding'],
      ['input', '[CLS] token + position embedding'], ['layer', 'LayerNorm'], ['layer', 'Bidirectional self-attention (no causal mask)'],
      ['layer', 'Residual add'], ['layer', 'LayerNorm'], ['layer', 'MLP (GELU)'], ['layer', 'Residual add'],
      ['output', 'Final LayerNorm'], ['output', 'Classification head on [CLS]'], ['output', 'Softmax over classes'],
    ],
  },
  {
    id: 'diffusion', title: 'Latent diffusion (U-Net / DiT)', status: 'template',
    outline: [
      ['input', 'Text prompt → text encoder → condition embeddings'], ['input', 'Gaussian noise latent z_T (inference) / VAE encoder → z₀ (training)'],
      ['input', 'Timestep embedding t'], ['layer', 'Denoiser block: self-attention over latent patches'],
      ['layer', 'Cross-attention to condition embeddings'], ['layer', 'MLP / ResNet block, modulated by t'],
      ['output', 'Predicted noise ε (or velocity v)'], ['output', 'Scheduler update z_t → z_(t−1), repeated for N steps'],
      ['output', 'VAE decoder → image'],
    ],
  },
  {
    id: 'multimodal', title: 'Multimodal (vision encoder + LLM)', status: 'template',
    outline: [
      ['input', 'Image → vision encoder (ViT) → patch features'], ['input', 'Projector (MLP) → visual tokens in LLM width H'],
      ['input', 'Text → tokenizer → token embeddings'], ['input', 'Interleaved sequence: visual tokens + text tokens'],
      ['layer', 'Decoder layer × L: causal self-attention over the joint sequence'], ['layer', 'FFN (dense or MoE)'],
      ['output', 'LM head → logits'], ['output', 'Softmax → next text token'],
    ],
  },
];

const LLAMA3_PAPER = 'https://arxiv.org/abs/2407.21783';
const DSV3_PAPER = 'https://arxiv.org/abs/2412.19437';

window.MODELS = [
  {
    id: 'llama-3.1-8b', family: 'llm',
    name: 'Llama 3.1 8B', org: 'Meta', year: 2024,
    why: 'Widely deployed open-weight dense model. 16 GB of BF16 weights fit one 80 GB GPU, so single-GPU memory and bandwidth limits appear without any parallelism.',
    source: 'https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json',
    layers: 32, hidden: 4096, vocab: 128256, context: 131072,
    attention: { kind: 'GQA', heads: 32, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 14336, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 500000, norm: 'RMSNorm', normEps: 1e-5, tied: false,
    tokenizer: 'byte-level BPE (tiktoken-based)',
    presets: {
      training: { gpu: 'h100', dp: 8, tp: 1, pp: 1, ep: 1, cp: 1, zero: 2, recompute: 'full', seq: 8192, mbs: 1, gbs: 64,
        note: 'Example layout (not published).' },
      inference: { gpu: 'h100', dp: 1, tp: 1, pp: 1, ep: 1, cp: 1, batch: 16, prompt: 2048, output: 512, wdtype: 'bf16',
        note: 'Example layout: one GPU.' },
    },
  },
  {
    id: 'llama-3.1-70b', family: 'llm',
    name: 'Llama 3.1 70B', org: 'Meta', year: 2024,
    why: 'Widely deployed open-weight dense model. 141 GB of BF16 weights exceed one 80 GB GPU, so inference needs tensor parallelism inside a node.',
    source: 'https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json',
    layers: 80, hidden: 8192, vocab: 128256, context: 131072,
    attention: { kind: 'GQA', heads: 64, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 28672, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 500000, norm: 'RMSNorm', normEps: 1e-5, tied: false,
    tokenizer: 'byte-level BPE (tiktoken-based)',
    presets: {
      training: { gpu: 'h100', dp: 4, tp: 8, pp: 4, ep: 1, cp: 1, zero: 1, recompute: 'none', seq: 8192, mbs: 1, gbs: 128,
        note: 'Example layout (not published).' },
      inference: { gpu: 'h100', dp: 1, tp: 4, pp: 1, ep: 1, cp: 1, batch: 32, prompt: 2048, output: 512, wdtype: 'bf16',
        note: 'Example layout: TP 4 inside one node.' },
    },
  },
  {
    id: 'llama-3.1-405b', family: 'llm',
    name: 'Llama 3.1 405B', org: 'Meta', year: 2024,
    why: 'Largest dense open-weight model: 811 GB of BF16 weights, more than one 8 × 80 GB node. Pre-trained on 16,384 H100 GPUs with 4D parallelism (TP, CP, PP, DP).',
    source: 'https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json',
    sourceNote: 'Architecture from the Llama 3 paper, Table 3 (8 KV heads). Some redistributed checkpoints duplicate KV heads to 16 for 16-way TP.',
    layers: 126, hidden: 16384, vocab: 128256, context: 131072,
    attention: { kind: 'GQA', heads: 128, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 53248, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 500000, norm: 'RMSNorm', normEps: 1e-5, tied: false,
    tokenizer: 'byte-level BPE (tiktoken-based)',
    presets: {
      training: { gpu: 'h100', dp: 128, tp: 8, pp: 16, ep: 1, cp: 1, zero: 1, recompute: 'full', seq: 8192, mbs: 1, gbs: 2048,
        note: `Published (Llama 3 paper, Table 4): 16,384 GPUs, TP 8 × CP 1 × PP 16 × DP 128, sequence 8,192, 16 sequences per DP rank. Micro-batch size, ZeRO stage and recompute are examples. ${LLAMA3_PAPER}` },
      inference: { gpu: 'h100', dp: 1, tp: 8, pp: 1, ep: 1, cp: 1, batch: 16, prompt: 2048, output: 512, wdtype: 'fp8',
        note: 'Example layout: FP8 weights on one 8-GPU node, TP 8.' },
    },
  },
  {
    id: 'mixtral-8x7b', family: 'llm',
    name: 'Mixtral 8x7B', org: 'Mistral AI', year: 2023,
    why: 'Coarse-grained MoE: 8 large experts, top-2 routing, no shared expert; 47B total / 13B active. Contrast with DeepSeek-V3\'s 256 small experts.',
    source: 'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json',
    layers: 32, hidden: 4096, vocab: 32000, context: 32768,
    attention: { kind: 'GQA', heads: 32, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'moe', experts: 8, shared: 0, topK: 2, expertInter: 14336, act: 'silu', gated: true,
      denseLayers: 0, router: 'softmax', routerBias: false },
    position: 'rope', ropeTheta: 1000000, norm: 'RMSNorm', normEps: 1e-5, tied: false,
    tokenizer: 'SentencePiece BPE with byte fallback',
    presets: {
      training: { gpu: 'h100', dp: 32, tp: 1, pp: 1, ep: 8, cp: 1, zero: 2, recompute: 'full', seq: 4096, mbs: 1, gbs: 128,
        note: 'Example layout (not published): 32 GPUs, EP 8, so each expert has 4 DP replicas for ZeRO-2 to shard over.' },
      inference: { gpu: 'h100', dp: 1, tp: 2, pp: 1, ep: 1, cp: 1, batch: 16, prompt: 2048, output: 512, wdtype: 'bf16',
        note: 'Example layout: TP 2.' },
    },
  },
  {
    id: 'deepseek-v3', family: 'llm',
    name: 'DeepSeek-V3', org: 'DeepSeek', year: 2024,
    why: 'MoE at scale: 671B total / 37B active, 256 fine-grained routed experts + 1 shared, MLA compressed KV cache, FP8 training. Trained with PP 16 × EP 64 on 2,048 GPUs.',
    source: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json',
    layers: 61, hidden: 7168, vocab: 129280, context: 163840,
    attention: { kind: 'MLA', heads: 128, qRank: 1536, kvRank: 512, nopeDim: 128, ropeDim: 64, vDim: 128 },
    ffn: { kind: 'moe', experts: 256, shared: 1, topK: 8, expertInter: 2048, act: 'silu', gated: true,
      denseLayers: 3, denseInter: 18432, router: 'sigmoid', routerBias: true,
      groups: 8, topkGroups: 4, routedScale: 2.5 },
    position: 'rope', ropeTheta: 10000, norm: 'RMSNorm', normEps: 1e-6, tied: false,
    tokenizer: 'byte-level BPE (DeepSeek)',
    trainPrecision: { moments: 'bf16', note: 'AdamW moments in BF16; master weights and gradients in FP32 (DeepSeek-V3 technical report).' },
    presets: {
      training: { gpu: 'h100', dp: 128, tp: 1, pp: 16, ep: 64, cp: 1, zero: 1, recompute: 'full', seq: 4096, mbs: 1, gbs: 15360,
        note: `Published (DeepSeek-V3 technical report): 2,048 H800 GPUs, PP 16, EP 64 across 8 nodes, ZeRO-1 DP; sequence 4K; batch up to 15,360. H100 used here in place of H800. The report recomputes RMSNorm and MLA up-projections selectively; full recompute here is the closest available option. ${DSV3_PAPER}` },
      inference: { gpu: 'h100', dp: 8, tp: 4, pp: 1, ep: 32, cp: 1, batch: 256, prompt: 4096, output: 1024, wdtype: 'fp8',
        note: `Published prefill unit (DeepSeek-V3 technical report): 32 GPUs, TP 4 with SP, DP 8, EP 32. The decode unit is 320 GPUs (TP 4, DP 80, EP 320). Batch and lengths are examples. ${DSV3_PAPER}` },
    },
  },
];
