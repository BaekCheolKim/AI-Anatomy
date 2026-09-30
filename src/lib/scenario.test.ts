import { describe, expect, it } from 'vitest';
import { getGpu } from '../data/gpus';
import { getModel } from '../data/registry';
import { DEEPSEEK_V3, GPT2_SMALL } from '../test/fixtures';
import { scenario } from '../test/scenarios';
import { DEFAULT_SCENARIO, validateScenario } from './scenario';

const LLAMA_8B = getModel('llama-3.1-8b');
const H100 = getGpu('h100');
const fields = (issues: ReturnType<typeof validateScenario>, level: string) =>
  issues.filter((i) => i.level === level).map((i) => i.field);

describe('validateScenario', () => {
  it('accepts the default scenario', () => {
    expect(validateScenario(LLAMA_8B, DEFAULT_SCENARIO, H100)).toEqual([]);
  });

  it('requires tp·pp·dp = gpuCount', () => {
    expect(fields(validateScenario(LLAMA_8B, { ...DEFAULT_SCENARIO, tp: 2 }, H100), 'error')).toContain('gpuCount');
  });

  it('requires TP to divide the query heads', () => {
    expect(fields(validateScenario(GPT2_SMALL, scenario({ tp: 8 }), H100), 'error')).toContain('tp');
  });

  it('explains KV-head replication when tp > kvHeads', () => {
    const issues = validateScenario(LLAMA_8B, scenario({ tp: 16 }), H100);
    expect(fields(issues, 'error')).toEqual([]);
    expect(fields(issues, 'info')).toContain('tp');
  });

  it('rejects EP on a dense model and invalid EP groups on an MoE model', () => {
    expect(fields(validateScenario(LLAMA_8B, scenario({ tp: 8, ep: 8 }), H100), 'error')).toContain('ep');
    expect(fields(validateScenario(DEEPSEEK_V3, scenario({ tp: 8, ep: 8 }), H100), 'error')).toEqual([]);
    expect(fields(validateScenario(DEEPSEEK_V3, scenario({ tp: 4, ep: 8 }), H100), 'error')).toContain('ep');
  });

  it('warns past the context window and on dtypes the GPU cannot compute in', () => {
    expect(fields(validateScenario(GPT2_SMALL, scenario({ seqLen: 1000, genTokens: 100 }), H100), 'warn')).toContain('seqLen');
    expect(fields(validateScenario(LLAMA_8B, scenario({ actDtype: 'fp8_e4m3' }), getGpu('a100')), 'warn')).toContain('actDtype');
  });
});
