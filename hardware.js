// GPU / node specifications. Peak FLOP/s are dense tensor-core figures (datasheets also list 2x
// "with sparsity"). Bandwidths are per direction, per GPU.
//
// hbm, hbmBW   HBM capacity (bytes) and bandwidth (bytes/s)
// flops        dense peak FLOP/s by compute dtype
// nvlink       GPU-to-GPU bandwidth inside a node        perNode  GPUs per node sharing NVLink
// nic          GPU-to-GPU bandwidth across nodes          host     CPU-to-GPU link used to load checkpoints
// sramPerSM    on-chip L1 / shared memory per SM
// assumed      fields that depend on the system build and are not in the GPU datasheet

const GiB = 2 ** 30;
const GH200_DATASHEET = 'https://dam-cdn.nvd.orangelogic.com/AssetLink/tkno03fa645gb7f33w60rf200d330420.pdf';

window.GPUS = [
  {
    id: 'gh200-96gb', name: 'GH200 96GB',
    hbm: 96 * GiB, hbmBW: 4e12,                      // 96 GB HBM3, up to 4 TB/s
    flops: { bf16: 990e12, fp8: 1979e12 },             // dense; datasheet lists 1,979 / 3,958 TFLOPS with sparsity
    sramPerSM: 256 * 1024,                             // Hopper SM: 256 KB combined L1 / shared memory
    nvlink: 450e9, nvlinkName: 'NVLink',               // 18 NVLink 4 links, 900 GB/s total
    host: 450e9, hostName: 'NVLink-C2C',               // Grace CPU ↔ Hopper GPU, 900 GB/s total
    perNode: 4,                                        // assumed: 4 GH200 per node, NVLink between them
    nic: 50e9, nicName: 'NIC 400 Gb/s',                // assumed: one 400 Gb/s NIC per GH200
    assumed: 'Node layout (4 GH200 per node, NVLink between them) and NIC bandwidth (400 Gb/s per GH200) depend on the system build; they are not in the datasheet.',
    source: GH200_DATASHEET,
  },
  // Reserved for other GPUs, left blank on purpose. To add one, copy the entry above and fill every field.
];
