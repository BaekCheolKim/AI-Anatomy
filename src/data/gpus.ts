/**
 * GPU catalog. Peak throughput is DENSE tensor-core FLOP/s (NVIDIA datasheets
 * headline the 2:4-sparsity figure, which is 2× these). Link bandwidth is per
 * direction (datasheets usually quote the bidirectional total, 2× these).
 */
import type { GpuSpec } from '../types';

const GiB = 2 ** 30;
const MiB = 2 ** 20;
const KiB = 2 ** 10;
const T = 1e12;
const GBps = 1e9;

const A100_DATASHEET = 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf';
const AMPERE_WP = 'https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf';
const H100_DATASHEET = 'https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet';
const HOPPER_WP = 'https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper';
const H200_PAGE = 'https://www.nvidia.com/en-us/data-center/h200/';
const ADA_WP = 'https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf';

export const GPUS: readonly GpuSpec[] = [
  {
    id: 'a100',
    label: 'A100 SXM 80GB',
    arch: 'Ampere (GA100)',
    memory: { kind: 'HBM2e', bytes: 80 * GiB, bandwidth: 2039 * GBps },
    peakFlops: { fp32: 19.5 * T, tf32: 156 * T, bf16: 312 * T, fp16: 312 * T, int8: 624 * T, int4: 1248 * T },
    sms: 108,
    sramPerSm: 192 * KiB,
    l2Bytes: 40 * MiB,
    links: { nvlink: 300 * GBps, pcie: 32 * GBps },
    sources: [
      { field: 'memory', url: A100_DATASHEET },
      { field: 'peakFlops', url: A100_DATASHEET },
      { field: 'links', url: A100_DATASHEET, note: 'NVLink 3: 600 GB/s total; PCIe Gen4 x16' },
      { field: 'sms', url: AMPERE_WP },
      { field: 'sramPerSm', url: AMPERE_WP },
      { field: 'l2Bytes', url: AMPERE_WP },
    ],
  },
  {
    id: 'h100',
    label: 'H100 SXM 80GB',
    arch: 'Hopper (GH100)',
    memory: { kind: 'HBM3', bytes: 80 * GiB, bandwidth: 3350 * GBps },
    peakFlops: { fp32: 67 * T, tf32: 494.7 * T, bf16: 989.4 * T, fp16: 989.4 * T, fp8_e4m3: 1978.9 * T, int8: 1978.9 * T },
    sms: 132,
    sramPerSm: 256 * KiB,
    l2Bytes: 50 * MiB,
    links: { nvlink: 450 * GBps, pcie: 64 * GBps },
    sources: [
      { field: 'memory', url: H100_DATASHEET, note: 'nvidia-smi reports 81,559 MiB usable' },
      { field: 'peakFlops', url: H100_DATASHEET },
      { field: 'links', url: H100_DATASHEET, note: 'NVLink 4: 900 GB/s total; PCIe Gen5 x16' },
      { field: 'sms', url: HOPPER_WP },
      { field: 'sramPerSm', url: HOPPER_WP },
      { field: 'l2Bytes', url: HOPPER_WP },
    ],
  },
  {
    id: 'h200',
    label: 'H200 SXM 141GB',
    arch: 'Hopper (GH100)',
    memory: { kind: 'HBM3e', bytes: 141 * GiB, bandwidth: 4800 * GBps },
    peakFlops: { fp32: 67 * T, tf32: 494.7 * T, bf16: 989.4 * T, fp16: 989.4 * T, fp8_e4m3: 1978.9 * T, int8: 1978.9 * T },
    sms: 132,
    sramPerSm: 256 * KiB,
    l2Bytes: 50 * MiB,
    links: { nvlink: 450 * GBps, pcie: 64 * GBps },
    sources: [
      { field: 'memory.bytes', url: H200_PAGE, approx: true, note: 'Marketed as 141 GB; the unit (GB vs GiB) is not stated precisely' },
      { field: 'memory.bandwidth', url: H200_PAGE },
      { field: 'peakFlops', url: H200_PAGE, note: 'Same GH100 compute as H100 SXM' },
      { field: 'links', url: H200_PAGE },
      { field: 'sms', url: HOPPER_WP },
      { field: 'sramPerSm', url: HOPPER_WP },
      { field: 'l2Bytes', url: HOPPER_WP },
    ],
  },
  {
    id: 'rtx4090',
    label: 'RTX 4090 24GB',
    arch: 'Ada Lovelace (AD102)',
    memory: { kind: 'GDDR6X', bytes: 24 * GiB, bandwidth: 1008 * GBps },
    peakFlops: { fp32: 82.6 * T, tf32: 82.6 * T, bf16: 165.2 * T, fp16: 165.2 * T, fp8_e4m3: 330.3 * T, int8: 660.6 * T },
    sms: 128,
    sramPerSm: 128 * KiB,
    l2Bytes: 72 * MiB,
    links: { nvlink: null, pcie: 32 * GBps },
    sources: [
      { field: 'memory', url: ADA_WP },
      { field: 'peakFlops', url: ADA_WP, note: '16-bit and FP8 figures are with FP32 accumulate; FP16 accumulate doubles them' },
      { field: 'peakFlops.fp8_e4m3', url: ADA_WP, approx: true },
      { field: 'links', url: ADA_WP, note: 'No NVLink; PCIe Gen4 x16' },
      { field: 'sms', url: ADA_WP },
      { field: 'sramPerSm', url: ADA_WP },
      { field: 'l2Bytes', url: ADA_WP },
    ],
  },
];

export const GPU_BY_ID: ReadonlyMap<string, GpuSpec> = new Map(GPUS.map((g) => [g.id, g]));

export const DEFAULT_GPU_ID = 'h100';

export function getGpu(id: string | null | undefined): GpuSpec {
  const g = (id && GPU_BY_ID.get(id)) || GPU_BY_ID.get(DEFAULT_GPU_ID);
  if (!g) throw new Error('GPU catalog is empty');
  return g;
}
