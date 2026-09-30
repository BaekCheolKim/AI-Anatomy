/**
 * The shape-symbol glossary. Tensor shapes are written with these symbols
 * (`['B', 'S+G', 'Nkv', 'D']`) and resolved here against a model + scenario.
 * A symbol that does not apply to a model (e.g. `Rkv` for a GQA model) resolves
 * to undefined, and using it in a shape is an error the contract tests catch.
 */
import type { ModelSpec, Scenario } from '../types';
import {
  attnDims,
  denseMlp,
  expertDataParallel,
  expertGroup,
  layerCounts,
  paramBreakdown,
  savedActElems,
  workingSetElems,
} from './calc';
import type { Env } from './expr';

export interface DimDef {
  label: string;
  of: (m: ModelSpec, s: Scenario) => number | undefined;
}

const moe = (m: ModelSpec) => (m.config.ffn.kind === 'moe' ? m.config.ffn : undefined);
const mla = (m: ModelSpec) => (m.config.attention.kind === 'MLA' ? m.config.attention : undefined);

export const DIMS = {
  // scenario
  B: { label: 'batch — sequences in flight (global)', of: (_m, s) => s.batch },
  S: { label: 'prompt / training sequence length', of: (_m, s) => s.seqLen },
  G: { label: 'tokens generated per sequence', of: (_m, s) => s.genTokens },
  mb: { label: 'micro-batches per step', of: (_m, s) => s.microBatches },
  Bmb: { label: 'micro-batch size per DP replica = B / (dp·mb)', of: (_m, s) => s.batch / (s.dp * s.microBatches) },
  N: { label: 'GPUs', of: (_m, s) => s.gpuCount },
  tp: { label: 'tensor-parallel degree', of: (_m, s) => s.tp },
  pp: { label: 'pipeline-parallel degree', of: (_m, s) => s.pp },
  dp: { label: 'data-parallel degree', of: (_m, s) => s.dp },
  ep: { label: 'expert-parallel degree', of: (_m, s) => s.ep },
  Eg: { label: 'expert group — ranks one layer’s experts are spread over (ep, or tp when ep = 1)', of: (m, s) => (moe(m) ? expertGroup(s) : undefined) },
  Edp: { label: 'expert data-parallel replicas = tp·dp / Eg', of: (m, s) => (moe(m) ? expertDataParallel(s) : undefined) },

  // model shape
  L: { label: 'decoder layers', of: (m) => m.config.layers },
  Ld: { label: 'layers with a dense FFN', of: (m) => layerCounts(m.config).dense },
  Lm: { label: 'layers with an MoE FFN', of: (m) => (moe(m) ? layerCounts(m.config).moe : undefined) },
  H: { label: 'hidden size (residual stream width)', of: (m) => m.config.hidden },
  V: { label: 'vocabulary (embedding rows)', of: (m) => m.config.vocab },
  Pos: { label: 'learned position-table rows', of: (m) => (m.config.position.kind === 'learned' ? m.config.maxPositions : undefined) },
  Nh: { label: 'query heads', of: (m) => m.config.attention.heads },
  Nkv: { label: 'K/V heads', of: (m) => attnDims(m.config.attention).kvHeads ?? undefined },
  D: { label: 'head dimension', of: (m) => (m.config.attention.kind === 'MLA' ? undefined : m.config.attention.headDim) },
  Dqk: { label: 'query/key head dimension', of: (m) => attnDims(m.config.attention).qkHeadDim },
  Dv: { label: 'value head dimension', of: (m) => attnDims(m.config.attention).vHeadDim },
  Rq: { label: 'MLA query latent rank', of: (m) => mla(m)?.qLoraRank ?? undefined },
  Rkv: { label: 'MLA KV latent rank (d_c)', of: (m) => mla(m)?.kvLoraRank },
  Dn: { label: 'MLA non-RoPE q/k head dim', of: (m) => mla(m)?.qkNopeHeadDim },
  Dr: { label: 'MLA decoupled RoPE key dim (d_r)', of: (m) => mla(m)?.qkRopeHeadDim },
  F: { label: 'dense FFN intermediate size', of: (m) => denseMlp(m.config)?.interDim },
  Fe: { label: 'expert FFN intermediate size', of: (m) => moe(m)?.expert.interDim },
  E: { label: 'routed experts per MoE layer', of: (m) => moe(m)?.routedExperts },
  Es: { label: 'shared experts per MoE layer', of: (m) => moe(m)?.sharedExperts },
  K: { label: 'experts selected per token (top-k)', of: (m) => moe(m)?.topK },

  // parameter counts
  P: { label: 'total parameters', of: (m) => paramBreakdown(m.config).total },
  Pd: { label: 'parameters outside routed experts', of: (m) => paramBreakdown(m.config).dense },
  Pe: { label: 'routed-expert parameters', of: (m) => (moe(m) ? paramBreakdown(m.config).expertsRouted : undefined) },
  Px: { label: 'parameters per expert', of: (m) => (moe(m) ? paramBreakdown(m.config).perExpert : undefined) },
  Pa: { label: 'active parameters per token', of: (m) => paramBreakdown(m.config).active },
  Nopt: {
    label: 'fp32 optimizer copies per parameter (AdamW m, v, + master weights unless training in fp32)',
    of: (_m, s) => (s.weightDtype === 'fp32' ? 2 : 3),
  },

  // activation accounting
  A: { label: 'saved activation elements per token per layer', of: (m, s) => savedActElems(m, s.recompute).value },
  W: { label: 'inference working-set elements per token', of: (m) => workingSetElems(m).value },
} satisfies Record<string, DimDef>;

export type DimSymbol = keyof typeof DIMS;

export function dimEnv(m: ModelSpec, s: Scenario): Env {
  const env: Record<string, number | undefined> = {};
  for (const [sym, def] of Object.entries(DIMS) as [DimSymbol, DimDef][]) env[sym] = def.of(m, s);
  return env;
}
