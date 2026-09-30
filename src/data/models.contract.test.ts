/**
 * Every file under src/data/models/ must pass these checks. They are what makes
 * "add one file, touch nothing else" safe.
 */
import { describe, expect, it } from 'vitest';
import type { ModelSpec } from '../types';
import { attnDims, layerCounts, paramBreakdown } from '../lib/calc';
import { MODEL_FILES, MODELS } from './registry';

function resolvePath(root: unknown, path: string): unknown {
  let cur: unknown = root;
  for (const key of path.split('.')) {
    if (cur === null || typeof cur !== 'object' || !Object.hasOwn(cur, key)) return undefined;
    cur = (cur as Record<string, unknown>)[key];
  }
  return cur;
}

const isPosInt = (n: number) => Number.isInteger(n) && n > 0;

describe('model registry', () => {
  it('discovers at least one model', () => {
    expect(MODELS.length).toBeGreaterThan(0);
  });

  it('names each file after its model id', () => {
    for (const [path, m] of MODEL_FILES) expect(path).toBe(`./models/${m.id}.ts`);
  });

  it('has unique ids', () => {
    expect(new Set(MODELS.map((m) => m.id)).size).toBe(MODELS.length);
  });
});

describe.each(MODELS.map((m) => [m.id, m] as const))('%s', (_id, m: ModelSpec) => {
  it('has the descriptive fields filled in', () => {
    expect(m.id).toMatch(/^[a-z0-9][a-z0-9.-]*$/);
    for (const f of [m.label, m.family, m.org, m.blurb, m.tokenizer.algorithm]) expect(f.trim()).not.toBe('');
    expect(m.released).toMatch(/^\d{4}-(0[1-9]|1[0-2])$/);
  });

  it('has a sane config', () => {
    const c = m.config;
    for (const n of [c.layers, c.hidden, c.vocab, c.maxPositions, c.attention.heads]) expect(isPosInt(n)).toBe(true);
    const d = attnDims(c.attention);
    expect(isPosInt(d.qkHeadDim) && isPosInt(d.vHeadDim)).toBe(true);
    if (d.kvHeads !== null) expect(c.attention.heads % d.kvHeads).toBe(0);
    if (c.ffn.kind === 'moe') {
      expect(c.ffn.topK).toBeLessThanOrEqual(c.ffn.routedExperts);
      expect(layerCounts(c).moe).toBeGreaterThan(0);
    }
    expect(m.tokenizer.vocab).toBeLessThanOrEqual(c.vocab);
    expect(c.norm.eps).toBeGreaterThan(0);
    expect(paramBreakdown(c).total).toBeGreaterThan(0);
  });

  it('cites a source for the config, and every source field exists', () => {
    expect(m.sources.some((s) => s.field === 'config' || s.field.startsWith('config.'))).toBe(true);
    for (const s of m.sources) {
      expect(resolvePath(m, s.field), s.field).toBeDefined();
      expect(Boolean(s.url || s.note), `${s.field} needs a url or note`).toBe(true);
    }
  });
});
