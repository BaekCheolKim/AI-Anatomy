// GPU and interconnect specifications (NVIDIA datasheets). Peak FLOP/s are dense tensor-core
// figures (datasheets also quote 2x "with sparsity"). Link bandwidths are per direction, per GPU.
//
// hbm:    HBM capacity, bytes          hbmBW:  HBM bandwidth, bytes/s
// flops:  dense peak by compute dtype  nvlink: GPU-to-GPU inside a node
// nic:    GPU-to-GPU across nodes (one InfiniBand NIC per GPU)
// pcie:   host-to-GPU link             sm / sramPerSM / l2: on-chip resources

const GiB = 2 ** 30;

window.GPUS = [
  {
    id: 'h100', name: 'H100 SXM 80GB',
    hbm: 80 * GiB, hbmBW: 3.35e12,
    flops: { bf16: 989e12, fp8: 1979e12 },
    sm: 132, sramPerSM: 256 * 1024, l2: 50 * 2 ** 20,
    nvlink: 450e9, nic: 50e9, nicName: 'InfiniBand NDR 400 Gb/s', pcie: 64e9, pcieName: 'PCIe Gen5 x16',
    source: 'https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet',
  },
  {
    id: 'h200', name: 'H200 SXM 141GB',
    hbm: 141 * GiB, hbmBW: 4.8e12,
    flops: { bf16: 989e12, fp8: 1979e12 },
    sm: 132, sramPerSM: 256 * 1024, l2: 50 * 2 ** 20,
    nvlink: 450e9, nic: 50e9, nicName: 'InfiniBand NDR 400 Gb/s', pcie: 64e9, pcieName: 'PCIe Gen5 x16',
    source: 'https://www.nvidia.com/en-us/data-center/h200/',
  },
  {
    id: 'a100', name: 'A100 SXM 80GB',
    hbm: 80 * GiB, hbmBW: 2.039e12,
    flops: { bf16: 312e12 },
    sm: 108, sramPerSM: 192 * 1024, l2: 40 * 2 ** 20,
    nvlink: 300e9, nic: 25e9, nicName: 'InfiniBand HDR 200 Gb/s', pcie: 32e9, pcieName: 'PCIe Gen4 x16',
    source: 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf',
  },
];

window.GPUS_PER_NODE = 8;
