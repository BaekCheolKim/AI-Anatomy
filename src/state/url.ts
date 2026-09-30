/**
 * The URL query string is the source of truth for the whole view. A pasted
 * link reproduces it exactly:
 *
 *   ?model=llama-3.1-8b&lens=memory&node=kv-cache&depth=2&b=8&s=4096&gpu=h100
 *
 * `model`, `lens` and `depth` are always written, so a link keeps meaning the
 * same view when models are added and the default changes. Scenario fields are
 * written only when they differ from DEFAULT_SCENARIO, to keep links short.
 * Anything unparseable falls back to its default, and the fallback is reported.
 */
import type { Depth, Dtype, LensId, Recompute, Scenario, Workload, ZeroStage } from '../types';
import { DEPTHS, DTYPES, LENS_IDS } from '../types';
import { GPU_BY_ID } from '../data/gpus';
import { getModel, MODEL_BY_ID } from '../data/registry';
import { DEFAULT_SCENARIO } from '../lib/scenario';

export interface ViewState {
  modelId: string;
  lens: LensId;
  depth: Depth;
  /** Selected node in the current lens graph. */
  nodeId: string | null;
  scenario: Scenario;
}

export const DEFAULT_LENS: LensId = 'architecture';
export const DEFAULT_DEPTH: Depth = 1;

type IntField = 'batch' | 'seqLen' | 'genTokens' | 'gpuCount' | 'tp' | 'pp' | 'dp' | 'ep' | 'microBatches';

/** Query key, minimum, maximum. */
const INT_KEYS: Readonly<Record<IntField, readonly [string, number, number]>> = {
  batch: ['b', 1, 2 ** 20],
  seqLen: ['s', 1, 2 ** 24],
  genTokens: ['g', 0, 2 ** 24],
  gpuCount: ['n', 1, 2 ** 17],
  tp: ['tp', 1, 1024],
  pp: ['pp', 1, 1024],
  dp: ['dp', 1, 2 ** 17],
  ep: ['ep', 1, 4096],
  microBatches: ['mb', 1, 4096],
};

const DTYPE_KEYS = { weightDtype: 'wd', kvDtype: 'kvd', actDtype: 'ad' } as const;

const WORKLOAD_CODES: Readonly<Record<Workload, string>> = { inference: 'inf', training: 'train' };
const RECOMPUTE: readonly Recompute[] = ['none', 'full'];
const ZERO: readonly ZeroStage[] = [0, 1, 2, 3];
const NODE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/;

export interface ParsedView {
  view: ViewState;
  /** Human-readable notes about values in the URL that were ignored. */
  notices: string[];
}

export function parseView(search: string): ParsedView {
  const q = new URLSearchParams(search);
  const notices: string[] = [];

  /** The value for `key` if it is one of `options` (compared as strings); otherwise `fallback`, noting why. */
  function pick<T extends string | number>(key: string, options: readonly T[], fallback: T, show = String(fallback)): T {
    const raw = q.get(key);
    if (raw === null) return fallback;
    const hit = options.find((o) => String(o) === raw);
    if (hit === undefined) notices.push(`Ignored ${key}=${raw}; using ${show}.`);
    return hit ?? fallback;
  }

  const rawModel = q.get('model');
  const model = getModel(rawModel);
  if (rawModel !== null && !MODEL_BY_ID.has(rawModel)) notices.push(`Unknown model "${rawModel}"; showing ${model.label}.`);

  const lens = pick('lens', LENS_IDS, DEFAULT_LENS);
  const depth = pick('depth', DEPTHS, DEFAULT_DEPTH);
  const rawNode = q.get('node');
  const nodeId = rawNode !== null && NODE_ID.test(rawNode) ? rawNode : null;
  if (rawNode !== null && nodeId === null) notices.push(`Ignored node=${rawNode}; nothing selected.`);

  const d = DEFAULT_SCENARIO;
  const s: Scenario = { ...d };
  for (const [field, [key, min, max]] of Object.entries(INT_KEYS) as [IntField, readonly [string, number, number]][]) {
    const raw = q.get(key);
    if (raw === null) continue;
    const n = /^\d+$/.test(raw) ? Number(raw) : NaN;
    if (n >= min && n <= max) s[field] = n;
    else notices.push(`Ignored ${key}=${raw}; using ${d[field]}.`);
  }
  // gpuCount is implied by the layout unless the URL says otherwise
  if (q.get(INT_KEYS.gpuCount[0]) === null) s.gpuCount = s.tp * s.pp * s.dp;

  for (const [field, key] of Object.entries(DTYPE_KEYS) as [keyof typeof DTYPE_KEYS, string][]) {
    s[field] = pick<Dtype>(key, DTYPES, d[field]);
  }
  s.gpuId = pick('gpu', [...GPU_BY_ID.keys()], d.gpuId);
  const codes = Object.values(WORKLOAD_CODES);
  const code = pick('w', codes, WORKLOAD_CODES[d.workload]);
  s.workload = (Object.keys(WORKLOAD_CODES) as Workload[]).find((k) => WORKLOAD_CODES[k] === code) ?? d.workload;
  s.recompute = pick('rc', RECOMPUTE, d.recompute);
  s.zeroStage = pick('z', ZERO, d.zeroStage);

  return { view: { modelId: model.id, lens, depth, nodeId, scenario: s }, notices };
}

/** Deterministic query string (leading "?"), defaults omitted except model, lens and depth. */
export function serializeView(v: ViewState): string {
  const q = new URLSearchParams();
  q.set('model', v.modelId);
  q.set('lens', v.lens);
  if (v.nodeId !== null) q.set('node', v.nodeId);
  q.set('depth', String(v.depth));

  const s = v.scenario;
  const d = DEFAULT_SCENARIO;
  if (s.workload !== d.workload) q.set('w', WORKLOAD_CODES[s.workload]);
  for (const [field, [key]] of Object.entries(INT_KEYS) as [IntField, readonly [string, number, number]][]) {
    if (field === 'gpuCount') continue;
    if (s[field] !== d[field]) q.set(key, String(s[field]));
  }
  // gpuCount is implied by tp·pp·dp unless it deliberately disagrees
  if (s.gpuCount !== s.tp * s.pp * s.dp) q.set(INT_KEYS.gpuCount[0], String(s.gpuCount));
  for (const [field, key] of Object.entries(DTYPE_KEYS) as [keyof typeof DTYPE_KEYS, string][]) {
    if (s[field] !== d[field]) q.set(key, s[field]);
  }
  if (s.gpuId !== d.gpuId) q.set('gpu', s.gpuId);
  if (s.recompute !== d.recompute) q.set('rc', s.recompute);
  if (s.zeroStage !== d.zeroStage) q.set('z', String(s.zeroStage));
  return `?${q.toString()}`;
}
