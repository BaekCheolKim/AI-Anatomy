import type { GpuSpec, ModelSpec, Scenario, ScenarioIssue } from '../types';
import { DEFAULT_GPU_ID } from '../data/gpus';
import { attnDims } from './calc';

export const DEFAULT_SCENARIO: Readonly<Scenario> = {
  workload: 'inference',
  batch: 8,
  seqLen: 4096,
  genTokens: 512,
  weightDtype: 'bf16',
  kvDtype: 'bf16',
  actDtype: 'bf16',
  gpuCount: 1,
  gpuId: DEFAULT_GPU_ID,
  tp: 1,
  pp: 1,
  dp: 1,
  ep: 1,
  microBatches: 1,
  recompute: 'none',
  zeroStage: 0,
};

/**
 * Problems with a scenario for a given model. Errors mean the numbers are not
 * meaningful (the layout cannot exist). Warnings mean it can run but the
 * estimate is rougher. Info explains a surprising consequence.
 */
export function validateScenario(m: ModelSpec, s: Scenario, gpu: GpuSpec): ScenarioIssue[] {
  const out: ScenarioIssue[] = [];
  const err = (field: keyof Scenario, message: string) => out.push({ level: 'error', field, message });
  const warn = (field: keyof Scenario, message: string) => out.push({ level: 'warn', field, message });
  const info = (field: keyof Scenario, message: string) => out.push({ level: 'info', field, message });
  const cfg = m.config;
  const d = attnDims(cfg.attention);

  for (const f of ['batch', 'seqLen', 'gpuCount', 'tp', 'pp', 'dp', 'ep', 'microBatches'] as const) {
    if (!Number.isInteger(s[f]) || s[f] < 1) err(f, `${f} must be a positive integer.`);
  }
  if (!Number.isInteger(s.genTokens) || s.genTokens < 0) err('genTokens', 'genTokens must be a non-negative integer.');

  if (s.tp * s.pp * s.dp !== s.gpuCount) {
    err('gpuCount', `tp·pp·dp = ${s.tp * s.pp * s.dp} but there are ${s.gpuCount} GPUs.`);
  }
  if (d.heads % s.tp !== 0) err('tp', `TP must divide the ${d.heads} query heads.`);
  if (d.kvHeads !== null && s.tp > d.kvHeads) {
    info('tp', `tp = ${s.tp} > ${d.kvHeads} KV heads: each KV head is replicated on ${s.tp / d.kvHeads} ranks.`);
  } else if (d.kvHeads !== null && d.kvHeads % s.tp !== 0) {
    warn('tp', `TP does not divide the ${d.kvHeads} KV heads evenly.`);
  }
  if (cfg.layers % s.pp !== 0) warn('pp', `${cfg.layers} layers do not split evenly into ${s.pp} stages.`);

  if (cfg.ffn.kind === 'moe') {
    const E = cfg.ffn.routedExperts;
    if (E % s.ep !== 0) err('ep', `EP must divide the ${E} routed experts.`);
    if ((s.tp * s.dp) % s.ep !== 0) err('ep', 'The EP group must fit within the TP×DP ranks of a stage (ep must divide tp·dp).');
  } else if (s.ep !== 1) {
    err('ep', 'This model has no experts; ep must be 1.');
  }

  const ctx = s.workload === 'inference' ? s.seqLen + s.genTokens : s.seqLen;
  if (ctx > cfg.maxPositions) warn('seqLen', `${ctx} tokens exceeds the ${cfg.maxPositions}-token context window.`);

  if (s.workload === 'training') {
    if (s.batch % (s.dp * s.microBatches) !== 0) {
      warn('batch', `Batch ${s.batch} does not divide into dp·microBatches = ${s.dp * s.microBatches} equal micro-batches.`);
    }
    if (s.pp > 1 && s.microBatches < s.pp) {
      info('microBatches', `Fewer micro-batches (${s.microBatches}) than stages (${s.pp}): the pipeline bubble dominates.`);
    }
  } else if (s.zeroStage > 0) {
    info('zeroStage', 'ZeRO applies to training only and is ignored for inference.');
  }

  if (gpu.peakFlops[s.actDtype] === undefined) {
    warn('actDtype', `${gpu.label} has no ${s.actDtype} tensor-core path; compute throughput is not meaningful.`);
  }
  if (s.gpuCount > 8) info('gpuCount', `${s.gpuCount} GPUs spans more than one 8-GPU node: traffic crosses the network.`);
  return out;
}
