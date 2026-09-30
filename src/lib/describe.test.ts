import { describe, expect, it } from 'vitest';
import type { SourceRef } from '../types';
import { getModel } from '../data/registry';
import { DEEPSEEK_V3, GPT2_SMALL } from '../test/fixtures';
import { describeConfig, describeTraining } from './describe';
import { isApprox, sourceFor } from './sources';

const LLAMA_8B = getModel('llama-3.1-8b');
const value = (facts: ReturnType<typeof describeConfig>, label: string) => facts.find((f) => f.label === label)?.value;

describe('describeConfig', () => {
  it('describes each attention and FFN variant from the config', () => {
    const llama = describeConfig(LLAMA_8B);
    expect(value(llama, 'Attention')).toBe('Grouped-query (GQA) · 32 query / 8 KV heads · head dim 128');
    expect(value(llama, 'FFN')).toBe('SwiGLU · intermediate 14,336');
    expect(value(llama, 'Positions')).toBe('RoPE · θ = 500,000 · Llama 3 frequency scaling ×8 from 8,192');

    const gpt2 = describeConfig(GPT2_SMALL);
    expect(value(gpt2, 'Attention')).toBe('Multi-head (MHA) · 12 heads · head dim 64');
    expect(value(gpt2, 'FFN')).toBe('GELU MLP · intermediate 3,072');
    expect(value(gpt2, 'Positions')).toBe('Learned absolute table · 1,024 positions');

    const ds = describeConfig(DEEPSEEK_V3);
    expect(value(ds, 'Attention')).toMatch(/^Multi-head latent \(MLA\) · 128 heads · KV latent 512 \+ RoPE key 64/);
    expect(value(ds, 'FFN')).toMatch(/256 routed \+ 1 shared · top-8 .* first 3 layers dense$/);
  });

  it('points every fact at a real path in the spec', () => {
    for (const m of [LLAMA_8B, GPT2_SMALL, DEEPSEEK_V3]) {
      for (const f of [...describeConfig(m), ...describeTraining(m)]) {
        const v = f.path.split('.').reduce<unknown>((o, k) => (o as Record<string, unknown> | undefined)?.[k], m);
        expect(v, f.path).toBeDefined();
      }
    }
  });
});

describe('sources', () => {
  const sources: SourceRef[] = [
    { field: 'config', url: 'a' },
    { field: 'config.attention', url: 'b' },
    { field: 'training.tokens', url: 'c', approx: true },
  ];

  it('finds the most specific source', () => {
    expect(sourceFor(sources, 'config.attention.kvHeads')?.url).toBe('b');
    expect(sourceFor(sources, 'config.layers')?.url).toBe('a');
    expect(sourceFor(sources, 'configx')).toBeUndefined();
  });

  it('marks a path approximate if it, a parent, or a child is flagged', () => {
    expect(isApprox(sources, 'training.tokens')).toBe(true);
    expect(isApprox(sources, 'training')).toBe(true);
    expect(isApprox(sources, 'training.gpuHours')).toBe(false);
    expect(isApprox(sources, 'config.layers')).toBe(false);
  });
});
