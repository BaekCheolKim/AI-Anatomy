/**
 * Tiny arithmetic language for tensor dimensions: `B`, `S+G`, `Nkv*D`, `H/tp`,
 * `min(pp,mb)`, `ceil(L/pp)`. Integers, identifiers, + - * /, parentheses and
 * the functions below. No other evaluation is possible, so dimension strings in
 * data files stay data.
 */

type Ast =
  | { t: 'num'; v: number }
  | { t: 'sym'; name: string }
  | { t: 'neg'; arg: Ast }
  | { t: 'bin'; op: '+' | '-' | '*' | '/'; l: Ast; r: Ast }
  | { t: 'call'; fn: FnName; args: Ast[] };

const FNS = {
  min: (...xs: number[]) => Math.min(...xs),
  max: (...xs: number[]) => Math.max(...xs),
  ceil: (x: number) => Math.ceil(x),
  floor: (x: number) => Math.floor(x),
} as const;
type FnName = keyof typeof FNS;

function isFn(name: string): name is FnName {
  return Object.hasOwn(FNS, name);
}

export class ExprError extends Error {}

type Token = { k: 'num'; v: number } | { k: 'id'; v: string } | { k: 'op'; v: string };

function tokenize(src: string): Token[] {
  const out: Token[] = [];
  const re = /(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|([-+*/(),])|\s+/y;
  while (re.lastIndex < src.length) {
    const at = re.lastIndex;
    const m = re.exec(src);
    if (!m) throw new ExprError(`Unexpected "${src[at]}" in "${src}"`);
    if (m[1] !== undefined) out.push({ k: 'num', v: Number(m[1]) });
    else if (m[2] !== undefined) out.push({ k: 'id', v: m[2] });
    else if (m[3] !== undefined) out.push({ k: 'op', v: m[3] });
  }
  return out;
}

function parse(src: string): Ast {
  const toks = tokenize(src);
  let p = 0;
  const peek = () => toks[p];
  const eat = (v: string) => {
    const tok = toks[p];
    if (tok?.k !== 'op' || tok.v !== v) throw new ExprError(`Expected "${v}" in "${src}"`);
    p++;
  };

  function expr(): Ast {
    let l = term();
    for (let tok = peek(); tok?.k === 'op' && (tok.v === '+' || tok.v === '-'); tok = peek()) {
      p++;
      l = { t: 'bin', op: tok.v, l, r: term() };
    }
    return l;
  }
  function term(): Ast {
    let l = factor();
    for (let tok = peek(); tok?.k === 'op' && (tok.v === '*' || tok.v === '/'); tok = peek()) {
      p++;
      l = { t: 'bin', op: tok.v, l, r: factor() };
    }
    return l;
  }
  function factor(): Ast {
    const tok = toks[p++];
    if (!tok) throw new ExprError(`Unexpected end of "${src}"`);
    if (tok.k === 'num') return { t: 'num', v: tok.v };
    if (tok.k === 'op' && tok.v === '-') return { t: 'neg', arg: factor() };
    if (tok.k === 'op' && tok.v === '(') {
      const inner = expr();
      eat(')');
      return inner;
    }
    if (tok.k === 'id') {
      const next = peek();
      if (next?.k === 'op' && next.v === '(') {
        if (!isFn(tok.v)) throw new ExprError(`Unknown function "${tok.v}" in "${src}"`);
        p++;
        const args = [expr()];
        for (let n = peek(); n?.k === 'op' && n.v === ','; n = peek()) {
          p++;
          args.push(expr());
        }
        eat(')');
        return { t: 'call', fn: tok.v, args };
      }
      return { t: 'sym', name: tok.v };
    }
    throw new ExprError(`Unexpected "${tok.v}" in "${src}"`);
  }

  const ast = expr();
  if (p !== toks.length) throw new ExprError(`Trailing input in "${src}"`);
  return ast;
}

const cache = new Map<string, Ast>();
function compiled(src: string): Ast {
  let ast = cache.get(src);
  if (!ast) {
    ast = parse(src);
    cache.set(src, ast);
  }
  return ast;
}

export type Env = Readonly<Record<string, number | undefined>>;

function run(ast: Ast, env: Env, src: string): number {
  switch (ast.t) {
    case 'num':
      return ast.v;
    case 'sym': {
      const v = env[ast.name];
      if (v === undefined) throw new ExprError(`"${ast.name}" is not defined for this model (in "${src}")`);
      return v;
    }
    case 'neg':
      return -run(ast.arg, env, src);
    case 'bin': {
      const l = run(ast.l, env, src);
      const r = run(ast.r, env, src);
      switch (ast.op) {
        case '+':
          return l + r;
        case '-':
          return l - r;
        case '*':
          return l * r;
        case '/':
          return l / r;
      }
    }
    case 'call': {
      const args = ast.args.map((a) => run(a, env, src));
      const fn: (...xs: number[]) => number = FNS[ast.fn];
      return fn(...args);
    }
  }
}

export function evalExpr(src: string, env: Env): number {
  return run(compiled(src), env, src);
}

/** Identifiers referenced by an expression, in first-use order (for "B = 8, S = 4096" glossaries). */
export function symbolsOf(src: string): string[] {
  const seen = new Set<string>();
  const walk = (a: Ast): void => {
    switch (a.t) {
      case 'sym':
        seen.add(a.name);
        return;
      case 'neg':
        walk(a.arg);
        return;
      case 'bin':
        walk(a.l);
        walk(a.r);
        return;
      case 'call':
        a.args.forEach(walk);
        return;
      case 'num':
        return;
    }
  };
  walk(compiled(src));
  return [...seen];
}

/** Display form: `Nkv*D` → `Nkv·D`. */
export function prettyExpr(src: string): string {
  return src.replace(/\s*\*\s*/g, '·');
}
