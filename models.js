// Model configs. Every value comes from the model's published config.json (linked in `source`).
// To add a model, append an entry. app.js builds the diagram and all numbers from these fields.
//
// attention.kind: 'MHA' | 'GQA' | 'MLA'      attention.bias: 'none' | 'qkv' | 'all'
// ffn.kind:       'dense' | 'moe'             ffn.act: 'silu' | 'gelu'   ffn.gated: SwiGLU-style gate
// position:       'learned' | 'rope'          norm: 'LayerNorm' | 'RMSNorm'

window.MODELS = [
  {
    id: 'gpt2-small',
    name: 'GPT-2 small',
    org: 'OpenAI', year: 2019,
    source: 'https://huggingface.co/openai-community/gpt2/blob/main/config.json',
    layers: 12, hidden: 768, vocab: 50257, context: 1024,
    attention: { kind: 'MHA', heads: 12, headDim: 64, bias: 'all' },
    ffn: { kind: 'dense', inter: 3072, act: 'gelu', gated: false, bias: true },
    position: 'learned', norm: 'LayerNorm', normEps: 1e-5, tied: true,
    tokenizer: 'byte-level BPE (GPT-2)',
  },
  {
    id: 'qwen2.5-7b',
    name: 'Qwen2.5 7B',
    org: 'Alibaba Qwen', year: 2024,
    source: 'https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json',
    layers: 28, hidden: 3584, vocab: 152064, context: 131072,
    attention: { kind: 'GQA', heads: 28, kvHeads: 4, headDim: 128, bias: 'qkv' },
    ffn: { kind: 'dense', inter: 18944, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 1000000, norm: 'RMSNorm', normEps: 1e-6, tied: false,
    tokenizer: 'byte-level BPE (Qwen)',
  },
  {
    id: 'llama-3.1-8b',
    name: 'Llama 3.1 8B',
    org: 'Meta', year: 2024,
    source: 'https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json',
    layers: 32, hidden: 4096, vocab: 128256, context: 131072,
    attention: { kind: 'GQA', heads: 32, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 14336, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 500000, norm: 'RMSNorm', normEps: 1e-5, tied: false,
    tokenizer: 'byte-level BPE (tiktoken-based)',
  },
  {
    id: 'mixtral-8x7b',
    name: 'Mixtral 8x7B',
    org: 'Mistral AI', year: 2023,
    source: 'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json',
    layers: 32, hidden: 4096, vocab: 32000, context: 32768,
    attention: { kind: 'GQA', heads: 32, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'moe', experts: 8, shared: 0, topK: 2, expertInter: 14336, act: 'silu', gated: true,
      denseLayers: 0, router: 'softmax', routerBias: false },
    position: 'rope', ropeTheta: 1000000, norm: 'RMSNorm', normEps: 1e-5, tied: false,
    tokenizer: 'SentencePiece BPE with byte fallback',
  },
  {
    id: 'llama-3.1-70b',
    name: 'Llama 3.1 70B',
    org: 'Meta', year: 2024,
    source: 'https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json',
    layers: 80, hidden: 8192, vocab: 128256, context: 131072,
    attention: { kind: 'GQA', heads: 64, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 28672, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 500000, norm: 'RMSNorm', normEps: 1e-5, tied: false,
    tokenizer: 'byte-level BPE (tiktoken-based)',
  },
  {
    id: 'deepseek-v3',
    name: 'DeepSeek-V3',
    org: 'DeepSeek', year: 2024,
    source: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json',
    layers: 61, hidden: 7168, vocab: 129280, context: 163840,
    attention: { kind: 'MLA', heads: 128, qRank: 1536, kvRank: 512, nopeDim: 128, ropeDim: 64, vDim: 128 },
    ffn: { kind: 'moe', experts: 256, shared: 1, topK: 8, expertInter: 2048, act: 'silu', gated: true,
      denseLayers: 3, denseInter: 18432, router: 'sigmoid', routerBias: true,
      groups: 8, topkGroups: 4, routedScale: 2.5 },
    position: 'rope', ropeTheta: 10000, norm: 'RMSNorm', normEps: 1e-6, tied: false,
    tokenizer: 'byte-level BPE (DeepSeek)',
  },
];
