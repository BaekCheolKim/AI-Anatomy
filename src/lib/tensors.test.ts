import { describe, expect, it } from 'vitest';
import { getGpu } from '../data/gpus';
import { getModel, MODELS } from '../data/registry';
import { DEEPSEEK_V3, FIXTURES, GPT2_SMALL, LLAMA_70B } from '../test/fixtures';
import { scenario, sweep } from '../test/scenarios';
import { BUDGET_BUCKETS, hbmBudget } from './budget';
import { bytesPerParam, kvBytes, paramsPerRank } from './calc';
import { dimEnv } from './dims';
import { evalExpr } from './expr';
import { perRankBytes, resolveShape, tensorBytes, tensorCatalog, tensorExists } from './tensors';

const LLAMA_8B = getModel('llama-3.1-8b');
const H100 = getGpu('h100');
const ALL = [...MODELS, ...FIXTURES];
const GiB = 2 ** 30;

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1);

describe.each(ALL.map((m) => [m.id, m] as const))('tensor catalog: %s', (_id, m) => {
  const catalog = tensorCatalog(m);

  it('has unique ids and shard dims that appear in the shape', () => {
    expect(new Set(catalog.keys()).size).toBe(catalog.size);
    for (const t of catalog.values()) {
      for (const sh of t.sharding ?? []) expect(t.shape, `${t.id} shards on ${sh.dim}`).toContain(sh.dim);
      expect(t.movement.length, t.id).toBeGreaterThan(10);
    }
  });

  it('resolves every shape to finite positive sizes across the scenario sweep', () => {
    for (const s of sweep(m)) {
      const env = dimEnv(m, s);
      for (const t of catalog.values()) {
        if (!tensorExists(t, s.workload)) continue;
        const dims = resolveShape(t, m, s);
        for (const [i, d] of dims.entries()) {
          expect(Number.isFinite(d) && d > 0, `${t.id}[${t.shape[i]}] = ${d}`).toBe(true);
        }
        const total = tensorBytes(t, m, s).value;
        const rank = perRankBytes(t, m, s).value;
        expect(rank).toBeGreaterThan(0);
        expect(rank).toBeLessThanOrEqual(total * (1 + 1e-12));
        for (const sh of t.sharding ?? []) if (sh.ways) expect(evalExpr(sh.ways, env)).toBeGreaterThan(0);
      }
    }
  });

  it('agrees with the named formula wherever one is attached', () => {
    for (const s of sweep(m)) {
      for (const t of catalog.values()) {
        if (!t.explain || !tensorExists(t, s.workload)) continue;
        expect(tensorBytes(t, m, s).value, t.id).toBeCloseTo(t.explain(m, s).value, 0);
      }
    }
  });

  it('budget buckets equal the per-rank parameter formulas', () => {
    for (const s of sweep(m)) {
      const b = hbmBudget(m, s, H100);
      const bw = bytesPerParam(s.weightDtype);
      expect(rel(b.buckets.weights.value, paramsPerRank(m, s, 3).value * bw)).toBeLessThan(1e-6);
      if (s.workload === 'training') {
        expect(rel(b.buckets.gradients.value, paramsPerRank(m, s, 2).value * bw)).toBeLessThan(1e-6);
        const copies = s.weightDtype === 'fp32' ? 2 : 3;
        expect(rel(b.buckets.optimizer.value, paramsPerRank(m, s, 1).value * 4 * copies)).toBeLessThan(1e-6);
        expect(b.buckets['kv-cache'].value).toBe(0);
      } else {
        expect(b.buckets.gradients.value).toBe(0);
        expect(b.buckets.optimizer.value).toBe(0);
      }
      const sum = BUDGET_BUCKETS.reduce((a, k) => a + b.buckets[k].value, 0);
      expect(rel(b.used.value, sum)).toBeLessThan(1e-12);
    }
  });
});

describe('KV cache sharding', () => {
  it('splits by KV head under TP, replicating heads once tp > kvHeads', () => {
    const t = tensorCatalog(LLAMA_8B).get('kv-cache');
    expect(t).toBeDefined();
    if (!t) return;
    const full = kvBytes(LLAMA_8B, scenario()).value;
    expect(perRankBytes(t, LLAMA_8B, scenario({ tp: 8 })).value).toBe(full / 8);
    // 8 KV heads over 16 ranks: each rank still holds one whole head
    expect(perRankBytes(t, LLAMA_8B, scenario({ tp: 16 })).value).toBe(full / 8);
    expect(perRankBytes(t, LLAMA_8B, scenario({ pp: 4 })).value).toBe(full / 4);
  });

  it('does not split an MLA latent over TP', () => {
    const t = tensorCatalog(DEEPSEEK_V3).get('kv-cache');
    expect(t).toBeDefined();
    if (!t) return;
    const s = scenario({ tp: 8, ep: 8 });
    expect(perRankBytes(t, DEEPSEEK_V3, s).value).toBe(tensorBytes(t, DEEPSEEK_V3, s).value);
  });

  it('shows the S² score matrix at its logical size', () => {
    const t = tensorCatalog(LLAMA_8B).get('attn-scores');
    expect(t?.location).toBe('GPU SRAM');
    if (!t) return;
    const s = scenario({ batch: 1, seqLen: 8192 });
    expect(tensorBytes(t, LLAMA_8B, s).value).toBe(32 * 8192 * 8192 * 4);
  });

  it('generates feature-specific tensors only where they apply', () => {
    expect(tensorCatalog(GPT2_SMALL).has('pos-table')).toBe(true);
    expect(tensorCatalog(LLAMA_8B).has('pos-table')).toBe(false);
    expect(tensorCatalog(LLAMA_8B).has('mlp-gate')).toBe(true);
    expect(tensorCatalog(GPT2_SMALL).has('mlp-gate')).toBe(false);
    expect(tensorCatalog(DEEPSEEK_V3).has('kv-latent')).toBe(true);
    expect(tensorCatalog(DEEPSEEK_V3).has('expert-weights')).toBe(true);
    expect(tensorCatalog(LLAMA_8B).has('expert-weights')).toBe(false);
  });
});

describe('HBM budget', () => {
  it('Llama 3.1 8B, B=8, S=4096+512, BF16 on one H100', () => {
    const b = hbmBudget(LLAMA_8B, scenario(), H100);
    expect(b.buckets.weights.value).toBe(8_030_261_248 * 2);
    expect(b.buckets['kv-cache'].value).toBe(4.5 * GiB);
    // working set 8·4096·51,200·2 + fp32 next-token logits 8·128256·4
    expect(b.buckets.activations.value).toBe(8 * 4096 * 51_200 * 2 + 8 * 128256 * 4);
    expect(b.used.value / GiB).toBeGreaterThan(24);
    expect(b.used.value / GiB).toBeLessThan(25.5);
    expect(b.fits).toBe(true);
    expect(b.used.approx).toBe(true);
    expect(b.buckets.weights.approx).toBe(false);
  });

  it('shows OOM when the scenario does not fit, and quantization rescues it', () => {
    const rtx = getGpu('rtx4090');
    expect(hbmBudget(LLAMA_8B, scenario({ gpuId: 'rtx4090' }), rtx).fits).toBe(false);
    expect(hbmBudget(LLAMA_8B, scenario({ gpuId: 'rtx4090', weightDtype: 'int4' }), rtx).fits).toBe(true);
    expect(hbmBudget(LLAMA_70B, scenario(), H100).fits).toBe(false);
    expect(hbmBudget(LLAMA_70B, scenario({ tp: 4 }), H100).fits).toBe(true);
  });

  it('training: AdamW state dominates until ZeRO shards it', () => {
    const one = hbmBudget(LLAMA_8B, scenario({ workload: 'training', batch: 1 }), H100);
    expect(one.buckets.optimizer.value).toBe(8_030_261_248 * 12);
    expect(one.fits).toBe(false);
    const zero3 = hbmBudget(LLAMA_8B, scenario({ workload: 'training', batch: 8, dp: 8, zeroStage: 3 }), H100);
    expect(zero3.buckets.optimizer.value).toBeCloseTo((8_030_261_248 * 12) / 8, 0);
    expect(zero3.fits).toBe(true);
  });

  it('full recompute trades activation memory for an extra forward', () => {
    const base = scenario({ workload: 'training', batch: 8, dp: 8, zeroStage: 3 });
    const none = hbmBudget(LLAMA_8B, base, H100).buckets.activations.value;
    const full = hbmBudget(LLAMA_8B, { ...base, recompute: 'full' }, H100).buckets.activations.value;
    expect(full).toBeLessThan(none / 4);
  });

  it('changing batch moves KV and activations, not weights', () => {
    const a = hbmBudget(LLAMA_8B, scenario({ batch: 8 }), H100);
    const b = hbmBudget(LLAMA_8B, scenario({ batch: 16 }), H100);
    expect(b.buckets.weights.value).toBe(a.buckets.weights.value);
    expect(b.buckets['kv-cache'].value).toBe(2 * a.buckets['kv-cache'].value);
    expect(b.buckets.activations.value).toBe(2 * a.buckets.activations.value);
  });
});
