/**
 * Test fixtures: real configs for the other seed models, used to exercise the
 * MHA / learned-position / MLA / MoE branches of calc before those models ship
 * as files under src/data/models/. Not registered, not shown in the UI.
 *
 * Each config was checked against its published config.json on 2026-09-30.
 */
import type { ModelSpec } from '../types';

/** https://huggingface.co/openai-community/gpt2/blob/main/config.json */
export const GPT2_SMALL: ModelSpec = {
  id: 'fixture-gpt2-small',
  label: 'GPT-2 small (fixture)',
  family: 'GPT-2',
  org: 'OpenAI',
  released: '2019-02',
  blurb: 'Fixture.',
  config: {
    layers: 12,
    hidden: 768,
    vocab: 50257,
    maxPositions: 1024,
    attention: { kind: 'MHA', heads: 12, headDim: 64, bias: true },
    ffn: { kind: 'dense', mlp: { interDim: 3072, gated: false, activation: 'gelu_tanh', bias: true } },
    position: { kind: 'learned' },
    norm: { kind: 'LayerNorm', eps: 1e-5 },
    tiedEmbeddings: true,
    nativeDtype: 'fp32',
  },
  tokenizer: { algorithm: 'Byte-level BPE', vocab: 50257 },
  sources: [{ field: 'config', url: 'https://huggingface.co/openai-community/gpt2/blob/main/config.json' }],
};

/** https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json */
export const LLAMA_70B: ModelSpec = {
  id: 'fixture-llama-3.1-70b',
  label: 'Llama 3.1 70B (fixture)',
  family: 'Llama 3.1',
  org: 'Meta',
  released: '2024-07',
  blurb: 'Fixture.',
  config: {
    layers: 80,
    hidden: 8192,
    vocab: 128256,
    maxPositions: 131072,
    attention: { kind: 'GQA', heads: 64, kvHeads: 8, headDim: 128, bias: false },
    ffn: { kind: 'dense', mlp: { interDim: 28672, gated: true, activation: 'silu', bias: false } },
    position: {
      kind: 'rope',
      theta: 500000,
      scaling: { kind: 'llama3', factor: 8, lowFreqFactor: 1, highFreqFactor: 4, originalMaxPositions: 8192 },
    },
    norm: { kind: 'RMSNorm', eps: 1e-5 },
    tiedEmbeddings: false,
    nativeDtype: 'bf16',
  },
  tokenizer: { algorithm: 'Byte-level BPE (tiktoken)', vocab: 128256 },
  sources: [{ field: 'config', url: 'https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json' }],
};

/** https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json */
export const DEEPSEEK_V3: ModelSpec = {
  id: 'fixture-deepseek-v3',
  label: 'DeepSeek-V3 (fixture)',
  family: 'DeepSeek-V3',
  org: 'DeepSeek',
  released: '2024-12',
  blurb: 'Fixture.',
  config: {
    layers: 61,
    hidden: 7168,
    vocab: 129280,
    maxPositions: 163840,
    attention: {
      kind: 'MLA',
      heads: 128,
      qLoraRank: 1536,
      kvLoraRank: 512,
      qkNopeHeadDim: 128,
      qkRopeHeadDim: 64,
      vHeadDim: 128,
    },
    ffn: {
      kind: 'moe',
      expert: { interDim: 2048, gated: true, activation: 'silu', bias: false },
      routedExperts: 256,
      sharedExperts: 1,
      topK: 8,
      router: { scoring: 'sigmoid', groups: 8, topkGroups: 4, selectionBias: true },
      leadingDense: { layers: 3, mlp: { interDim: 18432, gated: true, activation: 'silu', bias: false } },
    },
    position: {
      kind: 'rope',
      theta: 10000,
      scaling: { kind: 'yarn', factor: 40, originalMaxPositions: 4096, betaFast: 32, betaSlow: 1, mscale: 1, mscaleAllDim: 1 },
    },
    norm: { kind: 'RMSNorm', eps: 1e-6 },
    tiedEmbeddings: false,
    nativeDtype: 'fp8_e4m3',
  },
  tokenizer: { algorithm: 'Byte-level BPE', vocab: 129280 },
  sources: [{ field: 'config', url: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json' }],
};

export const FIXTURES: readonly ModelSpec[] = [GPT2_SMALL, LLAMA_70B, DEEPSEEK_V3];
