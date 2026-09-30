/**
 * Number formatting. Memory sizes use binary units (GiB) because that is how
 * capacity is sized (an "80 GB" H100 is 80 GiB). Throughput and FLOPs use
 * decimal SI units, as vendors quote them.
 */
import type { Derived, Dtype, Unit } from '../types';

const DTYPE_LABELS: Readonly<Record<Dtype, string>> = {
  int32: 'INT32',
  fp32: 'FP32',
  tf32: 'TF32',
  bf16: 'BF16',
  fp16: 'FP16',
  fp8_e4m3: 'FP8 E4M3',
  int8: 'INT8',
  int4: 'INT4',
};

export function dtypeLabel(d: Dtype): string {
  return DTYPE_LABELS[d];
}

const BIN = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'] as const;
const SI = ['', 'k', 'M', 'G', 'T', 'P', 'E'] as const;

/** Three significant figures for magnitudes ≥ 1. */
function sig(x: number): string {
  if (x === 0) return '0';
  const a = Math.abs(x);
  return x.toFixed(a >= 100 ? 0 : a >= 10 ? 1 : 2);
}

export function formatBytes(n: number): string {
  let i = 0;
  let v = n;
  while (Math.abs(v) >= 1024 && i < BIN.length - 1) {
    v /= 1024;
    i++;
  }
  return `${sig(v)} ${BIN[i]}`;
}

function formatSI(n: number, unit: string): string {
  if (Math.abs(n) >= 1e21) return `${formatSci(n)} ${unit}`;
  let i = 0;
  let v = n;
  while (Math.abs(v) >= 1000 && i < SI.length - 1) {
    v /= 1000;
    i++;
  }
  return `${sig(v)} ${SI[i]}${unit}`;
}

/** 7.23 × 10²³ */
export function formatSci(n: number): string {
  if (n === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(n)));
  const sup = String(e)
    .split('')
    .map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'['0123456789-'.indexOf(c)] ?? c)
    .join('');
  return `${(n / 10 ** e).toFixed(2)} × 10${sup}`;
}

export function formatFlops(n: number): string {
  return formatSI(n, 'FLOP');
}

/** 8.03 B, 671 B, 124 M. */
export function formatCount(n: number): string {
  if (Math.abs(n) < 1000) return Number.isInteger(n) ? String(n) : n.toFixed(2);
  const units = [
    [1e12, 'T'],
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'K'],
  ] as const;
  for (const [scale, suffix] of units) if (Math.abs(n) >= scale) return `${sig(n / scale)} ${suffix}`;
  return String(n);
}

/** 4,096 */
export function formatInt(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

export function formatValue(value: number, unit: Unit): string {
  switch (unit) {
    case 'bytes':
      return formatBytes(value);
    case 'bytes/s':
      return formatSI(value, 'B/s');
    case 'flops':
      return formatFlops(value);
    case 'flop/s':
      return formatSI(value, 'FLOP/s');
    case 'flop/byte':
      return `${value >= 1000 ? formatInt(value) : sig(value)} FLOP/B`;
    case 'params':
      return formatCount(value);
    case 'elements':
    case 'count':
      return Number.isInteger(value) ? formatInt(value) : sig(value);
    case 'ratio':
      return `${(value * 100).toFixed(1)}%`;
    case 'seconds':
      return value >= 1 ? `${sig(value)} s` : value >= 1e-3 ? `${sig(value * 1e3)} ms` : `${sig(value * 1e6)} µs`;
  }
}

export function formatDerived(d: Derived): string {
  return `${d.approx ? '≈ ' : ''}${formatValue(d.value, d.unit)}`;
}
