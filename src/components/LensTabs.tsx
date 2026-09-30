import { useRef, type KeyboardEvent } from 'react';
import { LENS_IDS } from '../types';
import { LENS_META } from '../lenses/meta';
import { useView } from '../state/store';
import { rovingIndex } from './roving';

export const LENS_PANEL_ID = 'lens-panel';

export function LensTabs() {
  const lens = useView((s) => s.lens);
  const setLens = useView((s) => s.setLens);
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: KeyboardEvent) => {
    const next = rovingIndex(e.key, LENS_IDS.indexOf(lens), LENS_IDS.length);
    const id = next === null ? undefined : LENS_IDS[next];
    if (next === null || !id) return;
    e.preventDefault();
    setLens(id);
    tabs.current[next]?.focus();
  };

  return (
    <div role="tablist" aria-label="Lens" onKeyDown={onKeyDown} className="flex gap-1 overflow-x-auto">
      {LENS_IDS.map((id, i) => {
        const selected = id === lens;
        return (
          <button
            key={id}
            ref={(el) => {
              tabs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={`tab-${id}`}
            aria-selected={selected}
            aria-controls={LENS_PANEL_ID}
            tabIndex={selected ? 0 : -1}
            onClick={() => setLens(id)}
            className={`shrink-0 rounded px-3 py-1 text-sm transition-colors ${
              selected ? 'bg-accent font-medium text-accent-ink' : 'text-muted hover:bg-raised hover:text-fg'
            }`}
          >
            {LENS_META[id].label}
          </button>
        );
      })}
    </div>
  );
}
