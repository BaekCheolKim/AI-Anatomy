import type { SourceRef } from '../types';

function covers(field: string, path: string): boolean {
  return path === field || path.startsWith(`${field}.`);
}

/** The most specific source backing `path` (a dotted path such as 'config.attention.kvHeads'). */
export function sourceFor(sources: readonly SourceRef[], path: string): SourceRef | undefined {
  let best: SourceRef | undefined;
  for (const s of sources) if (covers(s.field, path) && (!best || s.field.length > best.field.length)) best = s;
  return best;
}

/**
 * True if `path`, anything it sits under, or anything under it is flagged
 * approximate: a summary of 'training' is approximate if 'training.tokens' is.
 */
export function isApprox(sources: readonly SourceRef[], path: string): boolean {
  return sources.some((s) => s.approx === true && (covers(s.field, path) || covers(path, s.field)));
}
