/**
 * Per-GPU HBM budget. Built from the tensor catalog (not a parallel set of
 * formulas), so a bar in the Memory lens is exactly the sum of the per-rank
 * sizes of the tensors the other lenses show.
 */
import type { BudgetBucket, Derived, GpuSpec, HbmBudget, MemoryBucket, ModelSpec, Scenario } from '../types';
import { derive } from './calc';
import { perRankBytes, tensorCatalog, tensorExists } from './tensors';

export const MEMORY_BUCKETS: readonly MemoryBucket[] = ['weights', 'kv-cache', 'activations', 'gradients', 'optimizer'];
export const BUDGET_BUCKETS: readonly BudgetBucket[] = [...MEMORY_BUCKETS, 'runtime'];

/** CUDA context + library workspaces, per GPU. */
export const RUNTIME_RESERVED_BYTES = 1 * 2 ** 30;
/** Allocator fragmentation as a fraction of dynamic allocations. */
export const FRAGMENTATION = 0.05;

export function hbmBudget(m: ModelSpec, s: Scenario, gpu: GpuSpec): HbmBudget {
  const perBucket = new Map<MemoryBucket, Record<string, Derived>>(MEMORY_BUCKETS.map((b) => [b, {}]));
  for (const t of tensorCatalog(m).values()) {
    if (!t.bucket || !tensorExists(t, s.workload)) continue;
    const terms = perBucket.get(t.bucket);
    if (terms) terms[t.id] = perRankBytes(t, m, s);
  }

  const sums = {} as Record<MemoryBucket, Derived>;
  let dynamic = 0;
  for (const [b, terms] of perBucket) {
    const v = Object.values(terms).reduce((acc, d) => acc + d.value, 0);
    sums[b] = derive('bucketSum', v, terms);
    dynamic += v;
  }

  const runtime = derive('runtimeOverhead', RUNTIME_RESERVED_BYTES + FRAGMENTATION * dynamic, {
    C_ctx: RUNTIME_RESERVED_BYTES,
    f: FRAGMENTATION,
    U: dynamic,
  });
  const buckets: Record<BudgetBucket, Derived> = { ...sums, runtime };
  const used = derive('hbmUsed', dynamic + runtime.value, buckets);
  const capacity = derive('hbmCapacity', gpu.memory.bytes, { C_HBM: gpu.memory.bytes });
  return { buckets, used, capacity, fits: used.value <= capacity.value };
}
