import { LENS_META } from '../lenses/meta';
import { useModel, useView } from '../state/store';
import { LENS_PANEL_ID } from './LensTabs';

/** Graph canvas. Until lens graphs exist it shows which view the URL resolves to. */
export function Canvas() {
  const model = useModel();
  const lens = useView((s) => s.lens);
  const depth = useView((s) => s.depth);
  return (
    <section
      id={LENS_PANEL_ID}
      role="tabpanel"
      aria-labelledby={`tab-${lens}`}
      className="bg-grid relative flex min-h-[50vh] items-center justify-center border-x border-line p-8 lg:min-h-0"
    >
      <div className="max-w-md text-center">
        <p className="font-mono text-3xl tracking-[0.2em] text-muted uppercase">{LENS_META[lens].label}</p>
        <p className="mt-3 font-mono text-sm">
          {model.label} <span className="text-muted">·</span> depth {depth}
        </p>
        <p className="mt-4 text-sm text-muted">{LENS_META[lens].question}</p>
      </div>
    </section>
  );
}
