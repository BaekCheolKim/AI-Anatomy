import { describe, expect, it } from 'vitest';
import { evalExpr, ExprError, prettyExpr, symbolsOf } from './expr';

const env = { B: 8, S: 4096, G: 512, Nkv: 8, D: 128, tp: 16, pp: 4, mb: 2, H: 4096 };

describe('evalExpr', () => {
  it.each([
    ['B', 8],
    ['2', 2],
    ['S+G', 4608],
    ['Nkv*D', 1024],
    ['H/tp', 256],
    ['2*B*S', 65536],
    ['S-G*2', 3072],
    ['(S-G)*2', 7168],
    ['-B+10', 2],
    ['min(pp,mb)', 2],
    ['max(pp, mb, 3)', 4],
    ['ceil(Nkv/tp)', 1],
    ['floor(S / 1000)', 4],
    ['  B  ', 8],
  ])('%s = %d', (src, want) => {
    expect(evalExpr(src, env)).toBe(want);
  });

  it('rejects unknown symbols with a message naming them', () => {
    expect(() => evalExpr('Rkv+1', env)).toThrow(/"Rkv" is not defined/);
  });

  it('treats undefined env entries as not applicable', () => {
    expect(() => evalExpr('X', { X: undefined })).toThrow(ExprError);
  });

  it.each(['B+', '(B', 'B)', 'foo(B)', 'B $ S', 'B S', ''])('rejects malformed %j', (src) => {
    expect(() => evalExpr(src, env)).toThrow(ExprError);
  });
});

describe('symbolsOf / prettyExpr', () => {
  it('lists symbols in first-use order without duplicates or function names', () => {
    expect(symbolsOf('min(pp,mb)*L + L')).toEqual(['pp', 'mb', 'L']);
  });

  it('renders multiplication as a middle dot', () => {
    expect(prettyExpr('Nkv*D')).toBe('Nkv·D');
    expect(prettyExpr('2 * B')).toBe('2·B');
  });
});
