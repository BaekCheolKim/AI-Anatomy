import type { ModelSpec, Scenario } from '../types';
import { DEFAULT_SCENARIO } from '../lib/scenario';

export function scenario(over: Partial<Scenario> = {}): Scenario {
  const s = { ...DEFAULT_SCENARIO, ...over };
  return { ...s, gpuCount: over.gpuCount ?? s.tp * s.pp * s.dp };
}

const LAYOUTS: readonly Pick<Scenario, 'tp' | 'pp' | 'dp'>[] = [
  { tp: 1, pp: 1, dp: 1 },
  { tp: 8, pp: 1, dp: 1 },
  { tp: 2, pp: 4, dp: 1 },
  { tp: 1, pp: 1, dp: 8 },
  { tp: 4, pp: 2, dp: 4 },
  { tp: 16, pp: 1, dp: 1 }, // tp > kvHeads for 8-KV-head models
];

/**
 * A spread of scenarios covering both workloads, awkward sizes, every
 * precision role and the main parallel layouts. `ep` is set to the largest
 * valid value for MoE models so expert sharding is exercised too.
 */
export function sweep(m: ModelSpec): Scenario[] {
  const out: Scenario[] = [];
  const E = m.config.ffn.kind === 'moe' ? m.config.ffn.routedExperts : 1;
  for (const layout of LAYOUTS) {
    const group = layout.tp * layout.dp;
    const ep = E > 1 && E % group === 0 ? group : 1;
    for (const workload of ['inference', 'training'] as const) {
      out.push(scenario({ ...layout, ep, workload }));
      out.push(scenario({ ...layout, ep, workload, batch: 1, seqLen: 128, genTokens: 0 }));
      out.push(
        scenario({
          ...layout,
          ep,
          workload,
          batch: 64,
          seqLen: 8192,
          genTokens: 2048,
          weightDtype: 'int4',
          kvDtype: 'fp8_e4m3',
          actDtype: 'fp16',
          microBatches: 8,
          recompute: 'full',
          zeroStage: 3,
        }),
      );
      out.push(scenario({ ...layout, ep: 1, workload, weightDtype: 'fp32', actDtype: 'fp32', zeroStage: 1, microBatches: 4 }));
    }
  }
  return out;
}
