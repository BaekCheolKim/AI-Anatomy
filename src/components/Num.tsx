import type { Derived } from '../types';
import { FORMULAS } from '../lib/calc';
import { formatDerived } from '../lib/format';

/** The only way components display a computed number: it must be a Derived, so it always has a formula. */
export function Num({ value, className = '' }: { value: Derived; className?: string }) {
  const def = FORMULAS[value.formula];
  return (
    <span className={`font-mono tabular-nums ${className}`} title={value.approx ? `${def.name} (estimate)` : def.name}>
      {formatDerived(value)}
    </span>
  );
}

/** Small marker for a value whose source is not fully verified. */
export function ApproxMark({ note }: { note?: string }) {
  return (
    <abbr
      title={note ? `Approximate: ${note}` : 'Approximate: not fully verified against the source'}
      className="ml-1.5 rounded-sm border border-warn/50 px-1 align-middle font-mono text-[10px] text-warn no-underline"
    >
      approx
    </abbr>
  );
}
