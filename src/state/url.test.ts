import { describe, expect, it } from 'vitest';
import type { ViewState } from './url';
import { DEFAULT_DEPTH, DEFAULT_LENS, parseView, serializeView } from './url';
import { MODELS } from '../data/registry';
import { DEFAULT_SCENARIO } from '../lib/scenario';

const DEFAULT_MODEL = MODELS[0]?.id ?? '';

describe('parseView', () => {
  it('falls back to defaults for an empty query', () => {
    const { view, notices } = parseView('');
    expect(view).toEqual({
      modelId: DEFAULT_MODEL,
      lens: DEFAULT_LENS,
      depth: DEFAULT_DEPTH,
      nodeId: null,
      scenario: DEFAULT_SCENARIO,
    });
    expect(notices).toEqual([]);
  });

  it('reads the example link from the brief', () => {
    const { view, notices } = parseView('?model=deepseek-v3&lens=memory&node=kv-cache&depth=2&b=8&s=4096&gpu=h100');
    expect(view.lens).toBe('memory');
    expect(view.nodeId).toBe('kv-cache');
    expect(view.depth).toBe(2);
    expect(view.scenario).toMatchObject({ batch: 8, seqLen: 4096, gpuId: 'h100' });
    // DeepSeek-V3 is not registered yet: the link still opens, with a notice
    if (!MODELS.some((m) => m.id === 'deepseek-v3')) {
      expect(view.modelId).toBe(DEFAULT_MODEL);
      expect(notices[0]).toMatch(/Unknown model "deepseek-v3"/);
    }
  });

  it('ignores invalid values with a notice instead of failing', () => {
    const { view, notices } = parseView('?lens=nope&depth=7&b=-3&s=1e9&wd=fp64&gpu=tpu&w=x&rc=half&z=9&node=<script>');
    expect(view.lens).toBe(DEFAULT_LENS);
    expect(view.depth).toBe(DEFAULT_DEPTH);
    expect(view.nodeId).toBeNull();
    expect(view.scenario).toEqual(DEFAULT_SCENARIO);
    expect(notices).toHaveLength(10);
  });

  it('derives gpuCount from the layout unless given explicitly', () => {
    expect(parseView('?tp=8&dp=2').view.scenario.gpuCount).toBe(16);
    expect(parseView('?tp=8&n=4').view.scenario.gpuCount).toBe(4);
  });
});

describe('serializeView', () => {
  const base: ViewState = parseView('').view;

  it('always writes model, lens and depth; omits scenario defaults', () => {
    expect(serializeView(base)).toBe(`?model=${DEFAULT_MODEL}&lens=${DEFAULT_LENS}&depth=${DEFAULT_DEPTH}`);
  });

  it('round-trips every field', () => {
    const views: ViewState[] = [
      base,
      { ...base, lens: 'memory', depth: 3, nodeId: 'kv-cache' },
      {
        ...base,
        lens: 'parallelism',
        depth: 2,
        scenario: {
          ...DEFAULT_SCENARIO,
          workload: 'training',
          batch: 256,
          seqLen: 8192,
          genTokens: 0,
          weightDtype: 'fp8_e4m3',
          kvDtype: 'int8',
          actDtype: 'fp16',
          gpuId: 'a100',
          tp: 8,
          pp: 4,
          dp: 2,
          ep: 1,
          gpuCount: 64,
          microBatches: 16,
          recompute: 'full',
          zeroStage: 3,
        },
      },
      // gpuCount deliberately inconsistent with the layout survives too (validation reports it)
      { ...base, scenario: { ...DEFAULT_SCENARIO, tp: 2, gpuCount: 8 } },
    ];
    for (const v of views) {
      const { view, notices } = parseView(serializeView(v));
      expect(view).toEqual(v);
      expect(notices).toEqual([]);
    }
  });

  it('is deterministic', () => {
    const v = { ...base, scenario: { ...DEFAULT_SCENARIO, batch: 3, tp: 2, gpuCount: 2 } };
    expect(serializeView(v)).toBe(serializeView({ ...v }));
  });
});
