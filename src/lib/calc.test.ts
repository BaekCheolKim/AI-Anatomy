import { describe, expect, it } from 'vitest';
import type { Derived } from '../types';
import { DTYPES } from '../types';
import { getGpu } from '../data/gpus';
import { getModel } from '../data/registry';
import { DEEPSEEK_V3, GPT2_SMALL, LLAMA_70B } from '../test/fixtures';
import { scenario } from '../test/scenarios';
import {
  activeParams,
  allGatherBytes,
  allReduceBytes,
  allToAllBytes,
  attnFlops,
  bytesPerParam,
  decodeStep,
  derive,
  expertsTouched,
  FORMULAS,
  fwdFlops,
  kvBytes,
  matmulParams,
  paramBreakdown,
  paramsPerRank,
  paramsTotal,
  ppBubbleFraction,
  prefillStep,
  reduceScatterBytes,
  resolveDtype,
  ridgePoint,
  rooflineTime,
  savedActElems,
  tpShardedHidden,
  trainFlops,
  weightBytes,
  weightBytesRead,
  workingSetElems,
} from './calc';

const LLAMA_8B = getModel('llama-3.1-8b');
const H100 = getGpu('h100');
const GiB = 2 ** 30;

/** A nested derivation term. */
function sub(d: Derived, key: string): Derived {
  const t = d.terms[key];
  if (t === undefined || typeof t === 'number') throw new Error(`${d.formula}.${key} is not a Derived`);
  return t;
}

describe('bytesPerParam', () => {
  it.each([
    ['fp32', 4],
    ['tf32', 4],
    ['int32', 4],
    ['bf16', 2],
    ['fp16', 2],
    ['fp8_e4m3', 1],
    ['int8', 1],
    ['int4', 0.5],
  ] as const)('%s → %d', (d, b) => expect(bytesPerParam(d)).toBe(b));

  it('covers every dtype', () => {
    for (const d of DTYPES) expect(bytesPerParam(d)).toBeGreaterThan(0);
  });

  it('resolves dtype roles against the scenario', () => {
    const s = scenario({ weightDtype: 'int4', kvDtype: 'fp8_e4m3', actDtype: 'fp16' });
    expect(resolveDtype({ role: 'weight' }, s)).toBe('int4');
    expect(resolveDtype({ role: 'kv' }, s)).toBe('fp8_e4m3');
    expect(resolveDtype({ role: 'act' }, s)).toBe('fp16');
    expect(resolveDtype('int32', s)).toBe('int32');
  });
});

describe('parameter counts match published totals', () => {
  it('Llama 3.1 8B = 8,030,261,248 (published 8.03B)', () => {
    const p = paramBreakdown(LLAMA_8B.config);
    expect(p.total).toBe(8_030_261_248);
    expect(p.embedding).toBe(128256 * 4096);
    expect(p.lmHead).toBe(128256 * 4096);
    // per layer: q 4096² + k,v 2·4096·1024 + o 4096² = 41,943,040
    expect(p.attention).toBe(32 * 41_943_040);
    // per layer: 3 · 4096 · 14336
    expect(p.ffnDense).toBe(32 * 176_160_768);
    expect(p.norms).toBe(65 * 4096);
    expect(p.active).toBe(p.total);
  });

  it('Llama 3.1 70B = 70,553,706,496', () => {
    expect(paramBreakdown(LLAMA_70B.config).total).toBe(70_553_706_496);
  });

  it('GPT-2 small = 124,439,808 (tied head, learned positions, biases, LayerNorm)', () => {
    const p = paramBreakdown(GPT2_SMALL.config);
    expect(p.total).toBe(124_439_808);
    expect(p.lmHead).toBe(0);
    expect(p.positional).toBe(1024 * 768);
  });

  it('DeepSeek-V3 ≈ 671B total, ≈ 37B active', () => {
    const p = paramBreakdown(DEEPSEEK_V3.config);
    // Hand count: MLA 187,107,328/layer × 61; 3 dense FFN layers; 58 MoE layers of
    // 256 routed + 1 shared experts (44,040,192 each) + router (256·7168 + 256 bias).
    expect(p.total).toBe(671_026_419_200);
    expect(p.active).toBe(37_552_297_472);
    expect(Math.abs(p.total / 671e9 - 1)).toBeLessThan(0.001);
    expect(Math.abs(p.active / 37e9 - 1)).toBeLessThan(0.02);
    expect(p.perExpert).toBe(3 * 7168 * 2048);
  });

  it('moeActiveParams = dense + routed · k / E', () => {
    const a = activeParams(DEEPSEEK_V3);
    expect(a.formula).toBe('moeActiveParams');
    const t = a.terms as Record<string, number>;
    expect(a.value).toBeCloseTo((t.N_dense ?? 0) + ((t.N_routed ?? 0) * 8) / 256, 0);
    expect(activeParams(LLAMA_8B).formula).toBe('paramsTotal');
  });

  it('matmul params drop lookup tables but keep a tied LM head', () => {
    expect(matmulParams(LLAMA_8B).value).toBe(8_030_261_248 - 128256 * 4096);
    // GPT-2: the embedding is both a lookup (free) and the LM head (a matmul): net only the position table drops.
    expect(matmulParams(GPT2_SMALL).value).toBe(124_439_808 - 1024 * 768);
  });
});

describe('memory formulas', () => {
  it('weightBytes = params · bytesPerParam(weightDtype)', () => {
    expect(weightBytes(LLAMA_8B, scenario()).value).toBe(8_030_261_248 * 2);
    expect(weightBytes(LLAMA_8B, scenario({ weightDtype: 'int4' })).value).toBe(8_030_261_248 / 2);
  });

  it('kvBytesGQA = 2 · L · kvHeads · headDim · S · B · b', () => {
    const kv = kvBytes(LLAMA_8B, scenario({ batch: 8, seqLen: 4096, genTokens: 512 }));
    expect(kv.formula).toBe('kvBytesGQA');
    expect(kv.value).toBe(2 * 32 * 8 * 128 * 4608 * 8 * 2);
    expect(kv.value / GiB).toBe(4.5);
    // 128 KiB per token
    expect(kvBytes(LLAMA_8B, scenario(), { batch: 1, tokens: 1 }).value).toBe(128 * 1024);
  });

  it('MHA uses every head as a KV head', () => {
    expect(kvBytes(GPT2_SMALL, scenario(), { batch: 1, tokens: 1 }).value).toBe(2 * 12 * 12 * 64 * 2);
  });

  it('kvBytesMLA = L · (kvLoraRank + ropeHeadDim) · S · B · b', () => {
    const kv = kvBytes(DEEPSEEK_V3, scenario(), { batch: 1, tokens: 1 });
    expect(kv.formula).toBe('kvBytesMLA');
    expect(kv.value).toBe(61 * (512 + 64) * 2);
  });

  it('paramsPerRank splits dense params over tp·pp and ZeRO-shards over dp only in training', () => {
    const P = 8_030_261_248;
    expect(paramsPerRank(LLAMA_8B, scenario({ tp: 8 }), 3).value).toBe(P / 8);
    const train = scenario({ workload: 'training', dp: 8, zeroStage: 3 });
    expect(paramsPerRank(LLAMA_8B, train, 3).value).toBe(P / 8);
    expect(paramsPerRank(LLAMA_8B, { ...train, zeroStage: 2 }, 3).value).toBe(P);
    expect(paramsPerRank(LLAMA_8B, { ...train, workload: 'inference' }, 1).value).toBe(P);
  });

  it('paramsPerRank puts routed experts on the expert group', () => {
    const p = paramBreakdown(DEEPSEEK_V3.config);
    const ep8 = scenario({ tp: 8, ep: 8 });
    expect(paramsPerRank(DEEPSEEK_V3, ep8, 3).value).toBeCloseTo(p.dense / 8 + p.expertsRouted / 8, 0);
    const tpOnly = scenario({ tp: 8, ep: 1 });
    expect(paramsPerRank(DEEPSEEK_V3, tpOnly, 3).value).toBeCloseTo(p.dense / 8 + p.expertsRouted / 8, 0);
    // EP across tp×dp: experts split 64 ways, dense replicated over dp
    const wide = scenario({ tp: 8, dp: 8, ep: 64 });
    expect(paramsPerRank(DEEPSEEK_V3, wide, 3).value).toBeCloseTo(p.dense / 8 + p.expertsRouted / 64, 0);
  });

  it('saved activation elements per token per layer', () => {
    // Llama 8B: attn 2H + 2·Nh·D + 2·Nkv·D = 18,432; FFN 2H + 4F = 65,536
    expect(savedActElems(LLAMA_8B, 'none').value).toBe(83_968);
    expect(savedActElems(LLAMA_8B, 'full').value).toBe(4096);
    // GPT-2: 16·H, matching Korthikanti's 34·sbh bytes minus dropout masks and S² scores
    expect(savedActElems(GPT2_SMALL, 'none').value).toBe(16 * 768);
    expect(savedActElems(LLAMA_8B, 'none').approx).toBe(true);
  });

  it('inference working set per token', () => {
    // 2H + max(Q+K+V+O = 10,240, gate+up+product = 3F = 43,008)
    expect(workingSetElems(LLAMA_8B).value).toBe(8192 + 43_008);
  });
});

describe('FLOP formulas', () => {
  it('fwdFlops = 2 · N_mm · T; trainFlops = 6 · N_mm · T (8 with full recompute)', () => {
    const N = 8_030_261_248 - 128256 * 4096;
    expect(fwdFlops(LLAMA_8B, 1000).value).toBe(2 * N * 1000);
    expect(trainFlops(LLAMA_8B, 1000, 'none').value).toBe(6 * N * 1000);
    expect(trainFlops(LLAMA_8B, 1000, 'full').value).toBe(8 * N * 1000);
  });

  it('pretraining compute for Llama 3.1 8B is ≈ 7 × 10²³ FLOPs and marked approximate', () => {
    const t = LLAMA_8B.training;
    expect(t).toBeDefined();
    const f = trainFlops(LLAMA_8B, t?.tokens ?? 0, 'none', true);
    expect(f.value).toBeGreaterThan(6.5e23);
    expect(f.value).toBeLessThan(7.2e23);
    expect(f.approx).toBe(true);
  });

  it('attnFlops = 4 · B · H · S² · headDim per layer (the quadratic term)', () => {
    const S = 4096;
    const f = attnFlops(LLAMA_8B, 1, S, S);
    expect(f.value).toBe(32 * 4 * 1 * 32 * S * S * 128);
    // doubling S quadruples it
    expect(attnFlops(LLAMA_8B, 1, 2 * S, 2 * S).value).toBe(4 * f.value);
  });

  it('MLA attention uses d_qk = 192 and d_v = 128', () => {
    expect(attnFlops(DEEPSEEK_V3, 1, 10, 10).value).toBe(2 * 61 * 128 * 100 * (192 + 128));
  });
});

describe('roofline', () => {
  it('batch-1 decode on Llama 8B is ≈ 1 FLOP/byte: memory-bound', () => {
    const d = decodeStep(LLAMA_8B, scenario({ batch: 1, seqLen: 4096, genTokens: 512 }));
    const flops = 2 * (8_030_261_248 - 128256 * 4096) + 2 * 32 * 32 * 4608 * 256;
    const bytes = 8_030_261_248 * 2 + 2 * 32 * 8 * 128 * 4608 * 2;
    expect(d.flops.value).toBe(flops);
    expect(d.bytes.value).toBe(bytes);
    expect(d.intensity.value).toBeCloseTo(flops / bytes, 12);
    expect(d.intensity.value).toBeGreaterThan(0.9);
    expect(d.intensity.value).toBeLessThan(1.2);
    expect(d.intensity.formula).toBe('decodeIntensity');
  });

  it('decode intensity grows with batch, prefill intensity sits far above the ridge', () => {
    const ridge = ridgePoint(H100, 'bf16');
    expect(ridge?.value).toBeCloseTo(989.4e12 / 3.35e12, 6);
    const d1 = decodeStep(LLAMA_8B, scenario({ batch: 1 })).intensity.value;
    const d64 = decodeStep(LLAMA_8B, scenario({ batch: 64 })).intensity.value;
    expect(d64).toBeGreaterThan(10 * d1);
    const pre = prefillStep(LLAMA_8B, scenario({ batch: 1, seqLen: 4096 })).intensity.value;
    expect(pre).toBeGreaterThan(ridge?.value ?? Infinity);
  });

  it('ridgePoint is null when the GPU lacks the dtype', () => {
    expect(ridgePoint(getGpu('a100'), 'fp8_e4m3')).toBeNull();
  });

  it('rooflineTime is the max of compute time and memory time', () => {
    const F = fwdFlops(LLAMA_8B, 1);
    const Q = weightBytes(LLAMA_8B, scenario());
    const t = rooflineTime(F, Q, 1e15, 1e12);
    expect(t.value).toBe(Math.max(F.value / 1e15, Q.value / 1e12));
    expect(t.value).toBe(Q.value / 1e12);
  });

  it('MoE decode streams only the experts that were routed to', () => {
    const one = expertsTouched(DEEPSEEK_V3, 1);
    expect(one?.value).toBeCloseTo(8, 10);
    expect(expertsTouched(DEEPSEEK_V3, 100_000)?.value).toBeCloseTo(256, 6);
    expect(expertsTouched(LLAMA_8B, 1)).toBeNull();
    const s = scenario();
    const w1 = weightBytesRead(DEEPSEEK_V3, s, 1).value;
    const wAll = weightBytes(DEEPSEEK_V3, s).value;
    expect(w1).toBeLessThan(wAll / 10);
    expect(weightBytesRead(DEEPSEEK_V3, s, 1e6).value).toBeCloseTo(wAll, -3);
  });
});

describe('collectives', () => {
  it('allReduceBytes = 2 · (N−1)/N · M', () => {
    expect(allReduceBytes(1000, 4).value).toBe(1500);
    expect(allReduceBytes(1000, 1).value).toBe(0);
  });

  it('all-reduce = reduce-scatter + all-gather', () => {
    const M = 123_456_789;
    for (const n of [2, 4, 8, 64]) {
      expect(allReduceBytes(M, n).value).toBeCloseTo(reduceScatterBytes(M, n).value + allGatherBytes(M, n).value, 3);
      expect(allToAllBytes(M, n).value).toBeCloseTo(((n - 1) / n) * M, 3);
    }
  });
});

describe('parallelism', () => {
  it('tpShardedHidden = hidden / tp', () => {
    expect(tpShardedHidden(LLAMA_8B, scenario({ tp: 8 })).value).toBe(512);
  });

  it('ppBubbleFraction = (pp − 1) / microBatches', () => {
    expect(ppBubbleFraction(scenario({ pp: 4, microBatches: 8 })).value).toBe(3 / 8);
    expect(ppBubbleFraction(scenario({ pp: 1, microBatches: 1 })).value).toBe(0);
  });
});

describe('Derived', () => {
  it('records formula, unit and terms, and nests derivations', () => {
    const kv = kvBytes(LLAMA_8B, scenario());
    expect(kv.unit).toBe('bytes');
    expect(kv.terms).toMatchObject({ L: 32, N_kv: 8, D: 128, S: 4608, B: 8, b_kv: 2 });
    const d = decodeStep(LLAMA_8B, scenario()).intensity;
    expect(sub(sub(d, 'Q'), 'KV_read').formula).toBe('kvBytesGQA');
  });

  it('propagates approximation from nested terms', () => {
    const exact = derive('bucketSum', 1, { a: 1 });
    const est = derive('bucketSum', 1, { a: 1 }, true);
    expect(exact.approx).toBe(false);
    expect(derive('hbmUsed', 2, { exact, est }).approx).toBe(true);
    expect(derive('runtimeOverhead', 1, {}).approx).toBe(true);
  });

  it('every formula has a name, unit and KaTeX', () => {
    for (const [id, f] of Object.entries(FORMULAS)) {
      expect(f.name, id).not.toBe('');
      expect(f.tex, id).not.toBe('');
      expect(f.unit, id).toBeTruthy();
    }
  });

  it('paramsTotal terms add up to the total', () => {
    for (const m of [LLAMA_8B, GPT2_SMALL, DEEPSEEK_V3]) {
      const d = paramsTotal(m);
      const sum = Object.values(d.terms).reduce<number>((a, t) => a + (typeof t === 'number' ? t : t.value), 0);
      expect(sum).toBe(d.value);
    }
  });
});
