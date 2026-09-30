import { getGpu } from '../data/gpus';
import { hbmBudget } from '../lib/budget';
import { dtypeLabel, formatInt } from '../lib/format';
import { useModel, useView } from '../state/store';
import { Num } from './Num';

/** Read-only summary of the scenario from the URL. Editable controls come with the Memory lens. */
export function ScenarioBar() {
  const model = useModel();
  const s = useView((st) => st.scenario);
  const gpu = getGpu(s.gpuId);
  const budget = hbmBudget(model, s, gpu);
  const fill = Math.min(1, budget.used.value / budget.capacity.value);

  const items: [string, string][] = [
    ['workload', s.workload],
    ['batch', formatInt(s.batch)],
    [s.workload === 'training' ? 'seq' : 'prompt', formatInt(s.seqLen)],
    ...(s.workload === 'inference' ? ([['gen', formatInt(s.genTokens)]] as [string, string][]) : []),
    ['weights', dtypeLabel(s.weightDtype)],
    ['kv', dtypeLabel(s.kvDtype)],
    ['act', dtypeLabel(s.actDtype)],
    ['gpus', `${s.gpuCount}× ${gpu.label}`],
    ['tp·pp·dp·ep', `${s.tp}·${s.pp}·${s.dp}·${s.ep}`],
  ];

  return (
    <footer className="flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line bg-panel px-4 py-2">
      <span className="panel-label">Scenario</span>
      <dl className="flex flex-1 flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
        {items.map(([k, v]) => (
          <div key={k} className="flex gap-1.5">
            <dt className="text-muted">{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <div className="flex items-center gap-3 font-mono text-xs">
        <span className="text-muted">Σ HBM / GPU</span>
        <div
          role="meter"
          aria-label="HBM used per GPU"
          aria-valuemin={0}
          aria-valuemax={budget.capacity.value}
          aria-valuenow={Math.min(budget.used.value, budget.capacity.value)}
          className="h-2 w-28 overflow-hidden rounded-full bg-raised"
        >
          <div className={`h-full ${budget.fits ? 'bg-accent' : 'bg-danger'}`} style={{ width: `${fill * 100}%` }} />
        </div>
        <span>
          <Num value={budget.used} className={budget.fits ? '' : 'text-danger'} /> <span className="text-muted">/</span>{' '}
          <Num value={budget.capacity} />
        </span>
        {!budget.fits && <span className="font-semibold text-danger">OOM</span>}
      </div>
    </footer>
  );
}
