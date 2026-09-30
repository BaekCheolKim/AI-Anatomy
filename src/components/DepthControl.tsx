import { useRef, type KeyboardEvent } from 'react';
import { DEPTHS } from '../types';
import { useView } from '../state/store';
import { rovingIndex } from './roving';

export function DepthControl() {
  const depth = useView((s) => s.depth);
  const setDepth = useView((s) => s.setDepth);
  const radios = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: KeyboardEvent) => {
    const next = rovingIndex(e.key, DEPTHS.indexOf(depth), DEPTHS.length);
    const d = next === null ? undefined : DEPTHS[next];
    if (next === null || !d) return;
    e.preventDefault();
    setDepth(d);
    radios.current[next]?.focus();
  };

  return (
    <div className="flex items-center gap-2">
      <span id="depth-label" className="panel-label">
        Depth
      </span>
      <div
        role="radiogroup"
        aria-labelledby="depth-label"
        aria-keyshortcuts="1 2 3"
        onKeyDown={onKeyDown}
        className="flex overflow-hidden rounded border border-line-strong"
      >
        {DEPTHS.map((d, i) => {
          const checked = d === depth;
          return (
            <button
              key={d}
              ref={(el) => {
                radios.current[i] = el;
              }}
              type="button"
              role="radio"
              aria-checked={checked}
              aria-label={`Depth ${d}`}
              tabIndex={checked ? 0 : -1}
              onClick={() => setDepth(d)}
              className={`w-8 py-1 font-mono text-sm transition-colors ${
                checked ? 'bg-accent font-semibold text-accent-ink' : 'text-muted hover:bg-raised hover:text-fg'
              } ${i > 0 ? 'border-l border-line-strong' : ''}`}
            >
              {d}
            </button>
          );
        })}
      </div>
    </div>
  );
}
