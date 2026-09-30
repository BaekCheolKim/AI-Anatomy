/**
 * Llama 3.1 8B (base).
 *
 * Config: https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
 *   (verified against the ungated mirror https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json)
 * Paper:  The Llama 3 Herd of Models, arXiv:2407.21783 (Table 3; §3.1 tokenizer)
 * Card:   https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
 *
 * Cross-check: counting this config matrix by matrix gives 8,030,261,248
 * parameters, which matches the published "8.03B".
 */
import type { ModelSpec } from '../../types';

const CONFIG_URL = 'https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json';
const PAPER_URL = 'https://arxiv.org/abs/2407.21783';
const CARD_URL = 'https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md';

const model: ModelSpec = {
  id: 'llama-3.1-8b',
  label: 'Llama 3.1 8B',
  family: 'Llama 3.1',
  org: 'Meta',
  released: '2024-07',
  blurb:
    'The modern dense decoder in its plainest form: pre-norm RMSNorm, grouped-query attention (32 query heads sharing 8 KV heads), RoPE with Llama-3 frequency scaling to reach 128K context, and a SwiGLU MLP. Small enough to fit one GPU in BF16, so every cost is visible without parallelism in the way.',
  config: {
    layers: 32, //                 num_hidden_layers
    hidden: 4096, //               hidden_size
    vocab: 128256, //              vocab_size
    maxPositions: 131072, //       max_position_embeddings
    attention: {
      kind: 'GQA',
      heads: 32, //                num_attention_heads
      kvHeads: 8, //               num_key_value_heads
      headDim: 128, //             head_dim
      bias: false, //              attention_bias
    },
    ffn: {
      kind: 'dense',
      mlp: {
        interDim: 14336, //        intermediate_size
        gated: true, //            LlamaMLP: down(silu(gate(x)) * up(x))
        activation: 'silu', //     hidden_act
        bias: false, //            mlp_bias
      },
    },
    position: {
      kind: 'rope',
      theta: 500000, //            rope_theta
      scaling: {
        kind: 'llama3', //         rope_scaling.rope_type
        factor: 8,
        lowFreqFactor: 1,
        highFreqFactor: 4,
        originalMaxPositions: 8192,
      },
    },
    norm: { kind: 'RMSNorm', eps: 1e-5 }, // rms_norm_eps
    tiedEmbeddings: false, //      tie_word_embeddings
    nativeDtype: 'bf16', //        torch_dtype
  },
  tokenizer: {
    algorithm: 'Byte-level BPE (tiktoken)',
    vocab: 128256,
    note: '100K tokens from tiktoken plus 28K added for non-English languages, and 256 reserved special tokens.',
  },
  training: {
    tokens: 15e12,
    gpuHours: 1.46e6,
    gpu: 'H100-80GB',
  },
  sources: [
    { field: 'config', url: CONFIG_URL },
    { field: 'tokenizer', url: PAPER_URL, note: '§3.1: 100K tiktoken + 28K additional tokens' },
    { field: 'training.tokens', url: CARD_URL, approx: true, note: 'Model card says "~15 trillion tokens"' },
    { field: 'training.gpuHours', url: CARD_URL, note: 'Model card training-energy table: 1.46M GPU hours' },
    { field: 'training.gpu', url: CARD_URL },
  ],
};

export default model;
