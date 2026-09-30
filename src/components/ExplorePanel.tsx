import { LENS_META } from '../lenses/meta';
import { useView } from '../state/store';
import { Panel } from './Panel';

export function ExplorePanel() {
  const lens = useView((s) => s.lens);
  const meta = LENS_META[lens];
  return (
    <Panel title="Explore">
      <p className="font-medium">{meta.label}</p>
      <p className="mt-1 text-sm text-muted">{meta.question}</p>
      <p className="mt-6 text-xs text-muted">The node tree for this lens appears here once its graph is built.</p>
    </Panel>
  );
}
