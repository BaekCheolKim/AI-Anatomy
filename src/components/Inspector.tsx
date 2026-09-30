import type { Derived, ModelSpec } from '../types';
import { activeParams, matmulParams, paramsTotal } from '../lib/calc';
import { describeConfig, describeTraining, type Fact } from '../lib/describe';
import { isApprox, sourceFor } from '../lib/sources';
import { useModel, useView } from '../state/store';
import { ApproxMark, Num } from './Num';
import { Panel } from './Panel';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function released(ym: string): string {
  const [y, m] = ym.split('-');
  return `${MONTHS[Number(m) - 1] ?? ''} ${y ?? ''}`.trim();
}

function Facts({ model, facts }: { model: ModelSpec; facts: readonly Fact[] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
      {facts.map((f) => (
        <div key={f.label} className="contents">
          <dt className="text-muted">{f.label}</dt>
          <dd className="font-mono text-[13px]">
            {f.value}
            {isApprox(model.sources, f.path) && <ApproxMark note={sourceFor(model.sources, f.path)?.note} />}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function ParamRow({ label, value }: { label: string; value: Derived }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-line py-1.5 last:border-0">
      <span className="text-sm text-muted">{label}</span>
      <Num value={value} className="text-base" />
    </div>
  );
}

function ModelCard({ model }: { model: ModelSpec }) {
  const total = paramsTotal(model);
  const active = activeParams(model);
  const urls = [...new Set(model.sources.flatMap((s) => (s.url ? [s.url] : [])))];
  return (
    <div className="space-y-6">
      <div>
        <p className="text-lg font-semibold">{model.label}</p>
        <p className="font-mono text-xs text-muted">
          {model.org} · {released(model.released)}
        </p>
        <p className="mt-3 text-sm leading-relaxed">{model.blurb}</p>
      </div>

      <div>
        <h3 className="panel-label mb-1">Parameters</h3>
        <ParamRow label="Total" value={total} />
        {active.value !== total.value && <ParamRow label="Active per token" value={active} />}
        <ParamRow label="In matmuls (the N in 2·N·T)" value={matmulParams(model)} />
      </div>

      <div>
        <h3 className="panel-label mb-2">Config</h3>
        <Facts model={model} facts={describeConfig(model)} />
      </div>

      {model.training && (
        <div>
          <h3 className="panel-label mb-2">Training</h3>
          <Facts model={model} facts={describeTraining(model)} />
        </div>
      )}

      <div>
        <h3 className="panel-label mb-2">Sources</h3>
        <ul className="space-y-1 text-xs">
          {urls.map((u) => (
            <li key={u} className="truncate">
              <a href={u} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                {u.replace(/^https?:\/\//, '')}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export function Inspector() {
  const model = useModel();
  const nodeId = useView((s) => s.nodeId);
  const selectNode = useView((s) => s.selectNode);
  return (
    <Panel title="Inspector" className="lg:border-l-0">
      {nodeId !== null && (
        <div className="mb-6 rounded border border-line bg-raised p-3 text-sm">
          <p>
            Selected node <span className="font-mono text-accent">{nodeId}</span>
          </p>
          <p className="mt-1 text-xs text-muted">Node details appear here once this lens has a graph.</p>
          <button type="button" onClick={() => selectNode(null)} className="mt-2 text-xs text-muted hover:text-fg">
            Clear selection
          </button>
        </div>
      )}
      <ModelCard model={model} />
    </Panel>
  );
}
