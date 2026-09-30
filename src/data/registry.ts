/**
 * Model registry, discovered from the filesystem. Adding a model means adding
 * one file under ./models/ that default-exports a ModelSpec; nothing else
 * changes. Models are ordered by total parameter count.
 */
import type { ModelSpec } from '../types';
import { paramBreakdown } from '../lib/calc';

const modules = import.meta.glob<ModelSpec>('./models/*.ts', { eager: true, import: 'default' });

/** Module path → model, for the contract test that checks file name = id. */
export const MODEL_FILES: ReadonlyMap<string, ModelSpec> = new Map(Object.entries(modules));

export const MODELS: readonly ModelSpec[] = [...MODEL_FILES.values()].sort(
  (a, b) => paramBreakdown(a.config).total - paramBreakdown(b.config).total,
);

export const MODEL_BY_ID: ReadonlyMap<string, ModelSpec> = new Map(MODELS.map((m) => [m.id, m]));

/** Unknown or missing id falls back to the smallest model. */
export function getModel(id: string | null | undefined): ModelSpec {
  const m = (id && MODEL_BY_ID.get(id)) || MODELS[0];
  if (!m) throw new Error('No models found in src/data/models/');
  return m;
}
