// Target system and GPU specifications.
// Source of truth: HPC_spec_v1.md (KISTI-6 "HANGANG" pilot, measured 2026-10-06), plus the NVIDIA
// GH200 datasheet for values the system document does not list (peak FLOP/s, HBM bandwidth).
// Peak FLOP/s are dense tensor-core figures. Bandwidths are per direction.
//
// hbm, hbmBW   HBM capacity (bytes) and bandwidth (bytes/s)
// flops        dense peak FLOP/s by compute dtype
// perNode      GPUs per node
// nvlink       GPU-to-GPU inside a node, all links of one GPU (collectives)
// nvlinkPair   GPU-to-GPU inside a node, between one pair of GPUs (P2P send/recv)
// nic          GPU-to-GPU across nodes, per GPU
// host         CPU-to-GPU link (checkpoint path into HBM)
// sramPerSM    on-chip L1 / shared memory per SM
// assumed      values that neither source states

const MiB = 2 ** 20;
const HPC_SPEC = 'HPC_spec_v1.md';
const GH200_DATASHEET = 'https://dam-cdn.nvd.orangelogic.com/AssetLink/tkno03fa645gb7f33w60rf200d330420.pdf';

// Counts are a snapshot: the pilot system is expected to grow (HPC_spec_v1.md, pilot note).
window.SYSTEM = {
  name: 'KISTI-6 "HANGANG" pilot',
  platform: 'HPE Cray EX4000',
  gpuNodes: 22,
  gpus: 88,
  asOf: '2026-10-06',
  storage: 'Lustre /scratch (no node-local disk)',
  source: HPC_SPEC,
};

window.GPUS = [
  {
    id: 'gh200-96gb', name: 'GH200 96GB HBM3',
    hbm: 97871 * MiB,                                  // measured: nvidia-smi memory.total = 97871 MiB (≈ 95.6 GiB)
    hbmBW: 4e12,                                       // datasheet: up to 4 TB/s (HBM3 variant)
    flops: { bf16: 990e12, fp8: 1979e12 },             // datasheet, dense (1,979 / 3,958 TFLOPS with sparsity)
    sramPerSM: 256 * 1024,                             // Hopper SM: 256 KB combined L1 / shared memory
    perNode: 4,                                        // measured: 4 GH200 per node
    nvlink: 450e9, nvlinkPair: 150e9,                  // measured NV6 all-to-all: 6 links per pair × 25 GB/s; 18 links per GPU
    nvlinkName: 'NVLink (NV6 all-to-all)',
    host: 450e9, hostName: 'NVLink-C2C',               // Grace CPU ↔ Hopper GPU, 900 GB/s total
    hostMem: 120e9, hostMemName: 'LPDDR5X',            // measured: ~120 GB per Grace socket, 480 GB per node
    nic: 25e9, nicName: 'Slingshot 11 (200 Gb/s)',     // HPE Slingshot 11 (Cassini / CXI); 200 Gb/s per NIC
    assumed: 'One 200 Gb/s Slingshot 11 NIC per GH200: HPC_spec_v1.md names the fabric but not the NIC count per node.',
    source: [HPC_SPEC, GH200_DATASHEET],
  },
  // Reserved for other GPUs, left blank on purpose. To add one, copy the entry above and fill every field.
];
