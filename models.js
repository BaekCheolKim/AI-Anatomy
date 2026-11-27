// Models. Every architecture value comes from the model's published config.json (`source`).
// calc.js and the chapters build every number from these fields; nothing branches on a model id.
//
// attention.kind  'MHA' | 'GQA' | 'MLA'           attention.bias  'none' | 'qkv' | 'all'
// attention.qkNorm      RMSNorm on each query and key head (2 · headDim parameters per layer)
// attention.outputGate  q_proj also produces a sigmoid gate for the attention output (doubles q_proj)
// attention.sinks       one learned logit per head added to the softmax denominator
// attention.window      positions a 'sliding' layer attends to
// layerTypes            per layer: 'full' | 'sliding' | 'linear' (default: all 'full')
// linearAttention       Gated DeltaNet: key/value heads and dims, causal-conv width, state dtype bytes
// ffn.kind 'dense' | 'moe'; ffn.router: 'softmax-topk' | 'topk-softmax' | 'sigmoid-grouped'
// mtp                   multi-token-prediction modules stored in the checkpoint (not part of `total`)
// chat                  chat format (chapters/chat.js); chatBase: the post-trained model that uses it when this
//                       checkpoint is a base model; chatThink 'prompt': the generation prompt opens a <think> block
// layouts.rl            published RL recipe: prompts trained and generated per step, answers per prompt (G), prompt
//                       and total lengths, KL coefficient, clip range, inference-engine layout (gen)
// vision                ViT encoder whose output tokens enter the language model
// release               dtypes of the published weights, by tensor class (see calc.js weightClasses)
// checkpoint            safetensors element count reported by Hugging Face, and what it includes;
//                       params(m).checkpoint must equal it (minus stored non-parameter buffers)
// published             headline counts as the authors state them, with their convention

const pattern = (unit, times) => Array.from({ length: times }, () => unit).flat();

const LLAMA3_PAPER = 'https://arxiv.org/abs/2407.21783';
const LLAMA2_PAPER = 'https://arxiv.org/abs/2307.09288';
const DSV3_PAPER = 'https://arxiv.org/abs/2412.19437';
const LLAMA31_CARD = 'https://huggingface.co/meta-llama/Llama-3.1-405B';

window.MODELS = [
  {
    id: 'gpt-2', name: 'GPT-2 1.5B', short: 'GPT-2', org: 'OpenAI', year: 2019,
    postTraining: { text: 'None: released as a base model.' },
    why: 'The 2019 design that later models modify: multi-head attention, learned position embeddings, LayerNorm, a GELU MLP, and one matrix shared by the input embedding and the output layer.',
    source: 'https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json',
    checkpoint: { params: 1607942848, buffers: 48 * 1024 * 1024,
      note: 'The checkpoint also stores a 1,024 × 1,024 causal-mask buffer per layer (50,331,648 values); these are not parameters.' },
    published: { text: '1.5B parameters', total: 1.5e9 },
    layouts: { training: { dp: 8, tp: 1, pp: 1, ep: 1, cp: 1, gpusPerNode: 8, zero: 0, recompute: 'none', seq: 1024, mbs: 16, gbs: 512, example: true,
      text: 'Example layout: 8 GPUs, data parallel only, 512 sequences of 1,024 tokens per step.' } },
    layers: 48, hidden: 1600, vocab: 50257, context: 1024, tied: true,
    attention: { kind: 'MHA', heads: 25, kvHeads: 25, headDim: 64, bias: 'all' },
    ffn: { kind: 'dense', inter: 6400, act: 'gelu', gated: false, bias: true },
    position: 'learned', norm: 'LayerNorm', normEps: 1e-5,
    tokenizer: 'byte-level BPE', chat: null,
    release: { all: 'fp32' },
  },
  {
    id: 'llama-3.1-8b', name: 'Llama 3.1 8B', short: 'Llama 3.1 8B', org: 'Meta', year: 2024,
    postTraining: { text: 'Llama 3.1 8B Instruct: several rounds of supervised fine-tuning (including rejection-sampled data) and DPO (Llama 3 paper, section 4). Llama 3.1 8B itself is the pre-trained model.', url: LLAMA3_PAPER, methods: ['sft', 'dpo'] },
    why: 'Dense model with grouped-query attention; 16 GB of weights in BF16. An MLPerf Training benchmark.',
    source: 'https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json', gated: true,
    sourceNote: 'Layers, width, FFN width, heads, KV heads and RoPE θ from the Llama 3 paper, Table 3; vocabulary 128,256 and the llama3 RoPE scaling from Meta\'s llama-models reference code.',
    checkpoint: { params: 8030261248 },
    published: { text: '8B parameters', total: 8e9 },
    layouts: { training: { dp: 8, tp: 1, pp: 1, ep: 1, cp: 1, gpusPerNode: 8, zero: 1, recompute: 'none', seq: 8192, mbs: 1, gbs: 64, example: true,
      text: 'Example layout: 8 GPUs in one node, data parallel with ZeRO-1.' } },
    training: { tokens: 15e12, text: 'about 15 trillion tokens', url: LLAMA3_PAPER,
      gpuHours: 1.46e6, gpu: 'H100 80GB', gpuHoursText: '1.46M GPU hours (Llama 3.1 model card)', gpuHoursUrl: LLAMA31_CARD },
    layers: 32, hidden: 4096, vocab: 128256, context: 131072, tied: false,
    attention: { kind: 'GQA', heads: 32, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 14336, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 500000, norm: 'RMSNorm', normEps: 1e-5,
    ropeScaling: { type: 'llama3', factor: 8, lowFreqFactor: 1, highFreqFactor: 4, original: 8192 },
    tokenizer: 'byte-level BPE (tiktoken-based)', chat: 'llama3', chatBase: 'Llama 3.1 405B Instruct', chatBase: 'Llama 3.1 8B Instruct',
    release: { all: 'bf16' },
  },
  {
    id: 'llama-2-70b', name: 'Llama 2 70B', short: 'Llama 2 70B', org: 'Meta', year: 2023,
    postTraining: { text: 'Llama-2-chat: supervised fine-tuning, then RLHF with rejection sampling and PPO against separate helpfulness and safety reward models (Llama 2 paper). Llama 2 70B itself is the base model.', url: LLAMA2_PAPER, methods: ['sft', 'ppo'] },
    why: 'Dense model used for parameter-efficient fine-tuning (LoRA) in MLPerf Training.',
    source: 'https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json', gated: true,
    sourceNote: 'Width, layers, heads, KV heads, FFN width and vocabulary from Meta\'s llama-models reference parameters; RoPE θ 10,000 from the Llama 2 reference code; 4K context, GQA and SentencePiece BPE from the Llama 2 paper. The checkpoint count matches them exactly.',
    checkpoint: { params: 68976653312, buffers: 80 * 64, note: 'Includes 64 rotary inverse frequencies per layer (5,120 values), which are not parameters.' },
    published: { text: '70B parameters', total: 70e9, url: LLAMA2_PAPER },
    layouts: { training: { dp: 4, tp: 8, pp: 2, ep: 1, cp: 1, gpusPerNode: 8, zero: 1, recompute: 'full', seq: 4096, mbs: 1, gbs: 64, example: true,
      text: 'Example layout: 64 GPUs, TP 8 inside a node, PP 2, DP 4.' } },
    training: { tokens: 2e12, text: '2 trillion tokens', url: LLAMA2_PAPER,
      gpuHours: 1720320, gpu: 'A100 80GB', gpuHoursText: '1,720,320 GPU hours (Llama 2 paper, Table 2)', gpuHoursUrl: LLAMA2_PAPER },
    layers: 80, hidden: 8192, vocab: 32000, context: 4096, tied: false,
    attention: { kind: 'GQA', heads: 64, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 28672, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 10000, norm: 'RMSNorm', normEps: 1e-5,
    tokenizer: 'SentencePiece BPE', chat: 'llama2', chatBase: 'Llama 2 70B Chat',
    release: { all: 'fp16' },
  },
  {
    id: 'llama-3.1-405b', name: 'Llama 3.1 405B', short: 'Llama 3.1 405B', org: 'Meta', year: 2024,
    postTraining: { text: 'Llama 3.1 405B Instruct: several rounds of supervised fine-tuning (including rejection-sampled data) and DPO (Llama 3 paper, section 4). Llama 3.1 405B itself is the pre-trained model.', url: LLAMA3_PAPER, methods: ['sft', 'dpo'] },
    why: 'Dense model with 405B parameters, the largest Llama 3.1 model: 812 GB of weights in BF16. Pre-trained on 15.6T tokens with four parallel dimensions (TP, CP, PP, DP).',
    source: 'https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json', gated: true,
    sourceNote: 'Layers, width, FFN width, attention heads, 8 KV heads and RoPE θ from the Llama 3 paper, Table 3; vocabulary 128,256 (128,000 BPE tokens + 256 special tokens), RMSNorm ε and the llama3 RoPE scaling from Meta\'s llama-models reference code.',
    checkpoint: { params: 405853388800 },
    published: { text: '405B parameters', total: 405e9, url: LLAMA3_PAPER },
    training: { tokens: 15.6e12, text: '15.6 trillion tokens', url: LLAMA3_PAPER, flops: 3.8e25, flopsText: '3.8 × 10²⁵ FLOPs (Llama 3 paper)',
      gpuHours: 30.84e6, gpu: 'H100 80GB', gpuHoursText: '30.84M GPU hours (Llama 3.1 model card)', gpuHoursUrl: LLAMA31_CARD },
    layouts: {
      training: { dp: 128, tp: 8, pp: 16, ep: 1, cp: 1, gpusPerNode: 8, zero: 2, recompute: 'none', seq: 8192, mbs: 1, gbs: 2048,
        text: 'Published (Llama 3 paper, Table 4 and section 3.3.2): 16,384 GPUs in 8-GPU nodes, TP 8 × CP 1 × PP 16 × DP 128, sequence 8,192, 16 sequences per DP rank. Data parallelism is FSDP with optimizer states and gradients sharded and weights not resharded after the forward pass (closest setting here: ZeRO-2); no activation recomputation at 8K. Micro-batch size is an example. The page\'s activation estimate (≈) for this layout exceeds 80 GB; the paper does not give memory per GPU.', url: LLAMA3_PAPER },
    },
    layers: 126, hidden: 16384, vocab: 128256, context: 131072, tied: false,
    attention: { kind: 'GQA', heads: 128, kvHeads: 8, headDim: 128, bias: 'none' },
    ffn: { kind: 'dense', inter: 53248, act: 'silu', gated: true, bias: false },
    position: 'rope', ropeTheta: 500000, norm: 'RMSNorm', normEps: 1e-5,
    ropeScaling: { type: 'llama3', factor: 8, lowFreqFactor: 1, highFreqFactor: 4, original: 8192 },
    longContext: { text: 'Pre-trained at 8K, then the context was raised to 128K in six stages over about 800B tokens (Llama 3 paper, section 3.4.2).', url: LLAMA3_PAPER },
    tokenizer: 'byte-level BPE (tiktoken-based)', chat: 'llama3',
    release: { all: 'bf16' },
  },
  {
    id: 'deepseek-v3', name: 'DeepSeek-V3', short: 'DeepSeek-V3', org: 'DeepSeek', year: 2024,
    postTraining: { text: 'Supervised fine-tuning on 1.5M instances, including reasoning data generated by DeepSeek-R1 models, then reinforcement learning with GRPO using rule-based and model-based rewards (DeepSeek-V3 report, section 5).', url: DSV3_PAPER, methods: ['sft', 'grpo'] },
    why: 'MoE with 256 small routed experts and 1 shared expert, multi-head latent attention (a compressed KV cache), multi-token prediction and FP8 training. DeepSeek-R1 was trained from DeepSeek-V3-Base, the pre-trained checkpoint of this architecture, with cold-start supervised fine-tuning and GRPO reinforcement learning.',
    source: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json',
    checkpoint: { params: 684489845504, note: 'Includes the multi-token-prediction module (13.46B), which stores its own copies of the embedding and output matrices.' },
    published: { text: '671B total parameters, 37B activated for each token', total: 671e9, active: 37e9, url: DSV3_PAPER },
    training: { tokens: 14.8e12, text: '14.8 trillion tokens', url: DSV3_PAPER,
      gpuHours: 2.664e6, gpu: 'H800', gpuHoursText: '2.664M GPU hours for pre-training, 2.788M in total with context extension and post-training (DeepSeek-V3 report)', gpuHoursUrl: DSV3_PAPER },
    layouts: {
      training: { dp: 128, tp: 1, pp: 16, ep: 64, cp: 1, gpusPerNode: 8, zero: 1, recompute: 'full', seq: 4096, mbs: 1, gbs: 15360,
        text: 'Published (DeepSeek-V3 report): 2,048 GPUs in 8-GPU nodes, PP 16, EP 64 over 8 nodes, ZeRO-1 data parallelism, sequence 4K, batch up to 15,360 sequences. The report recomputes RMSNorm and the MLA up-projections selectively and keeps some activations in FP8; full recomputation is used here as the closest available option.', url: DSV3_PAPER },
      prefill: { tp: 4, dp: 8, ep: 32, gpus: 32, text: 'DeepSeek-V3 report: prefill unit of 32 GPUs, attention TP 4 with SP and DP 8, experts EP 32.', url: DSV3_PAPER },
      decode: { tp: 4, dp: 80, ep: 320, gpus: 320, text: 'DeepSeek-V3 report: decode unit of 320 GPUs, attention TP 4 with SP and DP 80, experts EP 320.', url: DSV3_PAPER },
    },
    layers: 61, hidden: 7168, vocab: 129280, context: 163840, tied: false,
    attention: { kind: 'MLA', heads: 128, qRank: 1536, kvRank: 512, nopeDim: 128, ropeDim: 64, vDim: 128 },
    ffn: { kind: 'moe', experts: 256, topK: 8, shared: 1, expertInter: 2048, act: 'silu', gated: true,
      denseLayers: 3, denseInter: 18432, router: 'sigmoid-grouped', routerBias: true, expertBias: false,
      groups: 8, topkGroups: 4, routedScale: 2.5 },
    mtp: { layers: 1, ownEmbedding: true },
    position: 'rope', ropeTheta: 10000, norm: 'RMSNorm', normEps: 1e-6,
    ropeScaling: { type: 'yarn', factor: 40, original: 4096, betaFast: 32, betaSlow: 1 },
    longContext: { text: 'Pre-trained at 4K, then extended with YaRN in two stages, to 32K and to 128K (DeepSeek-V3 report, section 4.3).', url: DSV3_PAPER },
    tokenizer: 'byte-level BPE', chat: 'deepseek',
    trainPrecision: { moments: 'bf16', note: 'AdamW moments in BF16; master weights and gradients in FP32 (DeepSeek-V3 report).' },
    release: { experts: 'fp8', linear: 'fp8' },
  },
  {
    id: 'qwen3-vl-235b', name: 'Qwen3-VL-235B-A22B', short: 'Qwen3-VL', org: 'Alibaba Qwen', year: 2025,
    postTraining: { text: 'Qwen3-VL technical report, section 4: supervised fine-tuning (non-thinking data for the Instruct edition, long chain-of-thought data for the Thinking edition), strong-to-weak distillation for the smaller models, then reinforcement learning with SAPO on rule-verifiable reasoning tasks and on general tasks.', url: 'https://arxiv.org/abs/2511.21631', methods: ['sft', 'rl'] },
    why: 'Vision-language MoE: a ViT encoder turns images into tokens for a 94-layer language model with 128 experts, 8 per token.',
    source: 'https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/blob/main/config.json',
    checkpoint: { params: 235670022896, note: 'Language model 235.09B + vision encoder 0.58B.' },
    published: { text: '235B total, 22B activated (A22B, in the name)', total: 235e9, active: 22e9, url: 'https://arxiv.org/abs/2511.21631' },
    training: { tokens: 38.2e12, text: '36 trillion tokens for the Qwen3 language model, then about 2.2 trillion multimodal tokens in four stages', url: 'https://arxiv.org/abs/2511.21631' },
    layouts: { training: { dp: 8, tp: 4, pp: 4, ep: 8, cp: 1, gpusPerNode: 8, zero: 1, recompute: 'full', seq: 8192, mbs: 1, gbs: 256, example: true,
      text: 'Example layout: 128 GPUs, TP 4, PP 4, EP 8, DP 8.' } },
    layers: 94, hidden: 4096, vocab: 151936, context: 262144, tied: false,
    attention: { kind: 'GQA', heads: 64, kvHeads: 4, headDim: 128, bias: 'none', qkNorm: true },
    ffn: { kind: 'moe', experts: 128, topK: 8, shared: 0, expertInter: 1536, act: 'silu', gated: true,
      denseLayers: 0, router: 'softmax-topk', normTopK: true, routerBias: false, expertBias: false },
    vision: { layers: 27, hidden: 1152, inter: 4304, heads: 16, patch: 16, temporalPatch: 2, merge: 2,
      channels: 3, posEmbed: 2304, outHidden: 4096, deepstack: 3 },
    position: 'rope', ropeTheta: 5000000, mrope: true, mropeSection: [24, 20, 20], norm: 'RMSNorm', normEps: 1e-6,
    tokenizer: 'byte-level BPE', chat: 'chatml',
    release: { all: 'bf16' },
  },
  {
    id: 'qwen3.5-397b', name: 'Qwen3.5-397B-A17B', short: 'Qwen3.5', org: 'Alibaba Qwen', year: 2026,
    postTraining: { text: 'Pre-training and post-training; the model card states "Reinforcement learning scaled across million-agent environments with progressively complex task distributions" and names no RL algorithm.', url: 'https://huggingface.co/Qwen/Qwen3.5-397B-A17B', methods: ['rl'] },
    why: 'Hybrid layers: 45 Gated DeltaNet layers (linear attention with a fixed-size state) and 15 gated full-attention layers. 512 experts, 10 routed + 1 shared per token; multi-token prediction; image input.',
    source: 'https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json',
    checkpoint: { params: 403397928944, note: 'Language model 396.35B + vision encoder 0.46B + multi-token-prediction module 6.60B.' },
    published: { text: '397B in total and 17B activated', total: 397e9, active: 17e9, url: 'https://huggingface.co/Qwen/Qwen3.5-397B-A17B' },
    layouts: {
      training: { dp: 4, tp: 8, pp: 8, ep: 32, cp: 1, gpusPerNode: 8, zero: 1, recompute: 'full', seq: 8192, mbs: 1, gbs: 256,
        text: 'NeMo-RL v0.8.0 GRPO recipe for this model (grpo-qwen3.5-397ba17b-32n8g-megatron.v2), the trainer\'s layout: 32 nodes × 8 GPUs, Megatron TP 8, PP 8, EP 32 (DP 4), sequence parallelism, activation recomputation, distributed optimizer (optimizer state sharded over data-parallel ranks: ZeRO-1); 256 sequences per step (32 prompts × 8 answers), micro-batch 1, sequence 8,192. The recipe sends tokens to experts with an all-gather dispatcher; the expert traffic below assumes all-to-all.', url: 'https://github.com/NVIDIA-NeMo/RL/blob/v0.8.0/examples/configs/recipes/llm/grpo-qwen3.5-397ba17b-32n8g-megatron.v2.yaml' },
      rl: { prompts: 32, genPrompts: 64, G: 8, prompt: 2048, total: 8192, kl: 0, clip: [0.2, 0.28], gen: { tp: 16, ep: 64 },
        text: 'the NeMo-RL v0.8.0 GRPO recipe for this model: each optimizer step trains on 32 prompts × 8 answers (256 sequences), prompts up to 2,048 tokens and prompt plus answer up to 8,192 (answers up to 6,144). It generates 64 prompts × 8 answers per batch and drops groups whose rewards are all equal (dynamic sampling). Its KL coefficient is 0, so no reference model is loaded, and it clips the probability ratio to [0.8, 1.28]. vLLM generates at TP 16 and EP 64', url: 'https://github.com/NVIDIA-NeMo/RL/blob/v0.8.0/examples/configs/recipes/llm/grpo-qwen3.5-397ba17b-32n8g-megatron.v2.yaml' } },
    layers: 60, hidden: 4096, vocab: 248320, context: 262144, tied: false,
    layerTypes: pattern(['linear', 'linear', 'linear', 'full'], 15),
    attention: { kind: 'GQA', heads: 32, kvHeads: 2, headDim: 256, bias: 'none', qkNorm: true, outputGate: true },
    linearAttention: { kind: 'Gated DeltaNet', keyHeads: 16, valueHeads: 64, keyDim: 128, valueDim: 128, conv: 4, stateBytes: 4 },
    ffn: { kind: 'moe', experts: 512, topK: 10, shared: 1, expertInter: 1024, sharedInter: 1024, sharedGate: true, normTopK: true,
      act: 'silu', gated: true, denseLayers: 0, router: 'softmax-topk', routerBias: false, expertBias: false },
    mtp: { layers: 1, ownEmbedding: false },
    vision: { layers: 27, hidden: 1152, inter: 4304, heads: 16, patch: 16, temporalPatch: 2, merge: 2,
      channels: 3, posEmbed: 2304, outHidden: 4096, deepstack: 0 },
    position: 'rope', ropeTheta: 10000000, ropeFraction: 0.25, mrope: true, mropeSection: [11, 11, 10], norm: 'RMSNorm', normEps: 1e-6, normGainPlusOne: true,
    tokenizer: 'byte-level BPE', chat: 'chatml', chatThink: 'prompt',
    release: { all: 'bf16' },
  },
  {
    id: 'gpt-oss-120b', name: 'gpt-oss-120b', short: 'gpt-oss', org: 'OpenAI', year: 2025,
    postTraining: { text: 'OpenAI describes a process similar to o4-mini: supervised fine-tuning and a high-compute reinforcement-learning stage, trained on the harmony format.', url: 'https://openai.com/index/introducing-gpt-oss/', methods: ['sft', 'rl'] },
    why: '128 experts stored in 4-bit MXFP4, 65 GB of weights in total. Layers alternate between 128-token sliding-window and full attention, with attention sinks. Trained on the harmony chat format, which has separate reasoning and answer channels.',
    source: 'https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json',
    checkpoint: { params: 116829156672 },
    published: { text: '117B parameters with 5.1B active parameters', total: 117e9, active: 5.1e9,
      convention: 'OpenAI\'s active count (5.13B) includes the output layer but leaves out the 0.58B-parameter embedding table (gpt-oss model card, arXiv:2508.10925, Table 1).', url: 'https://huggingface.co/openai/gpt-oss-120b' },
    layouts: { training: { dp: 8, tp: 1, pp: 1, ep: 8, cp: 1, gpusPerNode: 8, zero: 1, recompute: 'full', seq: 8192, mbs: 1, gbs: 64, example: true,
      text: 'Example layout: 8 GPUs in one node, EP 8, data parallel with ZeRO-1.' } },
    layers: 36, hidden: 2880, vocab: 201088, context: 131072, tied: false,
    layerTypes: pattern(['sliding', 'full'], 18),
    attention: { kind: 'GQA', heads: 64, kvHeads: 8, headDim: 64, bias: 'all', sinks: true, window: 128 },
    ffn: { kind: 'moe', experts: 128, topK: 4, shared: 0, expertInter: 2880, act: 'silu', gated: true, swigluLimit: 7,
      denseLayers: 0, router: 'topk-softmax', routerBias: true, expertBias: true },
    position: 'rope', ropeTheta: 150000, norm: 'RMSNorm', normEps: 1e-5,
    ropeScaling: { type: 'yarn', factor: 32, original: 4096, betaFast: 32, betaSlow: 1 },
    tokenizer: 'byte-level BPE (o200k_harmony)', chat: 'harmony',
    release: { experts: 'mxfp4' },
  },
];

// Models of other families, used in their own chapters (not in the model bar). Counts checked against
// the Hugging Face safetensors metadata, like the language models above.
window.OTHER_MODELS = {
  embedding: {
    id: 'qwen3-embedding-8b', name: 'Qwen3-Embedding-8B', org: 'Alibaba Qwen',
    source: 'https://huggingface.co/Qwen/Qwen3-Embedding-8B/blob/main/config.json', checkpoint: 7567295488,
    card: 'https://huggingface.co/Qwen/Qwen3-Embedding-8B',
    // A Qwen3 decoder without the output layer: the hidden state of the last token is the embedding.
    llm: { layers: 36, hidden: 4096, vocab: 151665, context: 40960, tied: false, noHead: true,
      attention: { kind: 'GQA', heads: 32, kvHeads: 8, headDim: 128, bias: 'none', qkNorm: true },
      ffn: { kind: 'dense', inter: 12288, act: 'silu', gated: true, bias: false }, position: 'rope', norm: 'RMSNorm' },
    dims: 4096, minDims: 32, maxLen: 32768,
  },
  diffusion: {
    id: 'flux.1-schnell', name: 'FLUX.1 [schnell]', org: 'Black Forest Labs',
    source: 'https://huggingface.co/black-forest-labs/FLUX.1-schnell', checkpoint: 11891178560,
    sourceNote: 'Configs are gated; the architecture below (diffusers FluxTransformer2DModel) reproduces the checkpoint count exactly.',
    transformer: { double: 19, single: 38, heads: 24, headDim: 128, inChannels: 64, jointDim: 4096, pooledDim: 768, mlpRatio: 4 },
    latentChannels: 16, vaeDown: 8, patch: 2, textTokens: 256, steps: 4,
  },
  speech: {
    id: 'whisper-large-v3', name: 'Whisper large-v3', org: 'OpenAI',
    source: 'https://huggingface.co/openai/whisper-large-v3/blob/main/config.json', checkpoint: 1543490560,
    d: 1280, encLayers: 32, decLayers: 32, heads: 20, ffn: 5120, mels: 128, vocab: 51866, srcPositions: 1500, tgtPositions: 448,
  },
};
