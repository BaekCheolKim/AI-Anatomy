/**
 * Guards for the rules the design depends on, checked on the source text:
 * - no model id outside src/data/models/ (so adding a model touches one file)
 * - no `any`
 * - only calc.ts may construct a Derived (so every displayed number has a formula)
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { MODELS } from './data/registry';

const SRC = fileURLToPath(new URL('.', import.meta.url));

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : /\.(ts|tsx)$/.test(name) ? [p] : [];
  });
}

const files = walk(SRC).map((abs) => ({
  path: relative(SRC, abs).split(sep).join('/'),
  text: readFileSync(abs, 'utf8'),
}));

const isTest = (p: string) => /\.test\.tsx?$/.test(p) || p.startsWith('test/');

describe('architecture', () => {
  it('mentions model ids only in their own data file', () => {
    const offenders: string[] = [];
    for (const f of files) {
      if (f.path.startsWith('data/models/') || isTest(f.path)) continue;
      for (const m of MODELS) {
        if (new RegExp(`['"\`]${m.id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"\`]`).test(f.text)) {
          offenders.push(`${f.path} mentions '${m.id}'`);
        }
      }
    }
    expect(offenders).toEqual([]);
  });

  it('uses no `any`', () => {
    const anyType = /(?::|\bas|<|,|\|)\s*any\b(?!\w)|\bany\[\]/;
    const offenders = files
      .filter((f) => f.path !== 'architecture.test.ts')
      .flatMap((f) =>
        f.text
          .split('\n')
          .map((line, i) => ({ line: line.replace(/\/\/.*$/, ''), n: i + 1 }))
          .filter(({ line }) => anyType.test(line))
          .map(({ n }) => `${f.path}:${n}`),
      );
    expect(offenders).toEqual([]);
  });

  it('constructs Derived values only in lib/calc.ts', () => {
    const offenders = files.filter((f) => f.path !== 'lib/calc.ts' && /\bas Derived\b/.test(f.text)).map((f) => f.path);
    expect(offenders).toEqual([]);
  });
});
