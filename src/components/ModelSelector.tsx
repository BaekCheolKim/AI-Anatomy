import { MODELS } from '../data/registry';
import { paramsTotal } from '../lib/calc';
import { formatDerived } from '../lib/format';
import { useView } from '../state/store';

export function ModelSelector() {
  const modelId = useView((s) => s.modelId);
  const setModel = useView((s) => s.setModel);
  return (
    <label className="flex items-center gap-2">
      <span className="panel-label">Model</span>
      <select
        value={modelId}
        onChange={(e) => setModel(e.target.value)}
        className="rounded border border-line-strong bg-raised px-2 py-1 font-mono text-sm text-fg hover:border-accent"
      >
        {MODELS.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label} · {formatDerived(paramsTotal(m))}
          </option>
        ))}
      </select>
    </label>
  );
}
