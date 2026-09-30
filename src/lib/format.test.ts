import { describe, expect, it } from 'vitest';
import { formatBytes, formatCount, formatFlops, formatSci, formatValue } from './format';

describe('format', () => {
  it('bytes in binary units', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(128 * 1024)).toBe('128 KiB');
    expect(formatBytes(4.5 * 2 ** 30)).toBe('4.50 GiB');
    expect(formatBytes(8_030_261_248 * 2)).toBe('15.0 GiB');
    expect(formatBytes(80 * 2 ** 30)).toBe('80.0 GiB');
  });

  it('FLOPs in SI units, scientific beyond zetta', () => {
    expect(formatFlops(1.5e12)).toBe('1.50 TFLOP');
    expect(formatFlops(989.4e12)).toBe('989 TFLOP');
    expect(formatFlops(6.75e23)).toBe('6.75 × 10²³ FLOP');
    expect(formatSci(1.2e-5)).toBe('1.20 × 10⁻⁵');
  });

  it('parameter counts', () => {
    expect(formatCount(8_030_261_248)).toBe('8.03 B');
    expect(formatCount(124_439_808)).toBe('124 M');
    expect(formatCount(671_026_419_200)).toBe('671 B');
    expect(formatCount(42)).toBe('42');
  });

  it('other units', () => {
    expect(formatValue(3 / 8, 'ratio')).toBe('37.5%');
    expect(formatValue(0.0123, 'seconds')).toBe('12.3 ms');
    expect(formatValue(295.34, 'flop/byte')).toBe('295 FLOP/B');
    expect(formatValue(27620.4, 'flop/byte')).toBe('27,620 FLOP/B');
    expect(formatValue(3.35e12, 'bytes/s')).toBe('3.35 TB/s');
    expect(formatValue(83968, 'elements')).toBe('83,968');
  });
});
