# AI-Anatomy

A static web page that shows how specific large models compute, where their tensors reside, and how data moves between GPUs during training and inference. All values are computed from published model configs and a measured hardware specification. The page runs no model.

## Opening it

Double-click `index.html`, or open it in any browser. There is nothing to install, build or serve; the page also works from `file://`.

## What the page shows

The view is **model × phase × lens**, chosen from the bar at the top and the controls under the title.

**Models.** Each one is included for a specific technique or scale:

| Model | Shows |
|---|---|
| Llama 3.1 8B | Dense GQA decoder; BF16 weights (16 GB) fit one GPU |
| Llama 3.1 70B | Dense; BF16 weights (141 GB) need tensor parallelism inside a node |
| Llama 3.1 405B | Largest dense open-weight model (811 GB BF16); 4D-parallel pre-training on 16,384 GPUs |
| Mixtral 8x7B | Coarse-grained MoE: 8 experts, top-2, no shared expert |
| DeepSeek-V3 | MoE at scale (671B total / 37B active, 256 routed + 1 shared expert), MLA compressed KV cache |

ViT, latent diffusion and multimodal (vision encoder + LLM) are listed as **templates**: their stage outlines are shown, but configs and numbers are not written yet.

**Phases.**
- **Training:** forward pass, cross-entropy loss, backward pass, gradient synchronization, AdamW optimizer step.
- **Inference:** no gradients or optimizer states. It is split into
  - **prefill:** all prompt positions in one pass;
  - **decode:** one new position per step, reading the KV cache.

**Lenses.**
- **Data flow:** every stage from tokenizer to next-token selection, generated from the model's config (MHA / GQA / MLA attention, dense or MoE FFN).
  - **Collective markers:** purple boxes appear where the current parallel layout requires communication, for example a TP all-reduce after the attention output projection, or an EP all-to-all around the experts.
  - **Navigation:** click a box, or step with Prev / Next or the ← → keys. The current box is outlined in solid blue and the next in dashed amber.
  - **Side panel:** shows the stage's operation, tensor shapes for the current scenario, per-GPU residency (HBM or on-chip SRAM), its collectives, and links to related stages.
- **Memory:** per-GPU HBM use, split into what is allocated at load (weights, FP32 master weights, AdamW moments) and what is materialized per step or per request (gradients, saved activations, KV cache, workspaces). Also shown:
  - transfers per GPU: checkpoint load, and HBM traffic per step;
  - in inference, KV-cache growth against sequence length, and a prefill vs decode comparison of arithmetic intensity against the GPU's ridge point (compute-bound vs memory-bandwidth-bound).
- **Parallelism:** one card per dimension: DP (with ZeRO-1/2/3 and FSDP), TP (with sequence parallelism), PP, EP and CP. Each card states what is sharded and what is replicated, and lists its collectives:
  - the operation: all-reduce, all-gather, reduce-scatter, all-to-all or P2P;
  - message size and count per step, volume per GPU;
  - the link it runs over (NVLink inside a node, network between nodes) and its time at that link's bandwidth.

**Scenario controls.** DP, TP, PP, EP and CP degrees, ZeRO stage, activation recompute, sequence length and batch sizes for training; batch, prompt and output length and weight dtype (BF16 / FP8) for inference. Each model has a preset per phase; the note under the controls says which preset values are published (with the paper) and which are examples. Warnings appear when a layout is invalid or exceeds the target system, for example a TP group spanning nodes.

**Deep-dive pages.** Links such as "FlashAttention →", "ZeRO stages →" and "Expert parallelism →" open `technique.html?id=…`. These pages are templates: section headings only, content not written yet.

The URL hash stores the full view (model, phase, step, lens, selected stage, scenario), so a copied link reopens the same view.

## Target hardware

The GPU is fixed to **NVIDIA GH200 96GB HBM3** in the node layout of the **KISTI-6 "HANGANG" pilot** (HPE Cray EX4000). Values marked *measured* come from that system's hardware specification (`HPC_spec_v1.md`, measured 2026-10-06; kept locally, not part of this repository). The others come from NVIDIA's GH200 datasheet.

| Value | Used | Source |
|---|---|---|
| HBM per GPU | 97,871 MiB (95.6 GiB) | measured |
| HBM bandwidth | 4 TB/s | datasheet |
| Dense peak | BF16 990 TFLOPS, FP8 1,979 TFLOPS | datasheet |
| GPUs per node | 4 | measured |
| NVLink inside a node | all-to-all, 6 links per pair: 150 GB/s per direction per pair, 450 GB/s per GPU across all links | measured topology |
| Between nodes | HPE Slingshot 11, 200 Gb/s (25 GB/s) per GPU | fabric measured; one NIC per GPU is assumed |
| CPU ↔ GPU | NVLink-C2C, 450 GB/s per direction | datasheet |
| Storage | Lustre, no node-local disk | measured |
| System size | 88 GPUs in 22 nodes (pilot snapshot, will grow) | measured |

Other GPUs are not listed; `hardware.js` has a blank slot for adding them.

## How the numbers are computed

All numbers come from `calc.js`; nothing is typed in by hand.

- **Parameters** are counted matrix by matrix from each config (attention, FFN or experts, router, norms, embeddings, LM head). The totals match the published counts, for example Llama 3.1 8B = 8,030,261,248 and DeepSeek-V3 = 671.0B total / 37.6B active.
- **Training memory** uses BF16 weights, FP32 gradient buffers, FP32 master weights and FP32 AdamW moments (BF16 moments for DeepSeek-V3, per its report). ZeRO-1 / 2 / 3 shard optimizer states, then also gradients, then also weights across the data-parallel group. Saved activations are counted per token per layer, assuming FlashAttention (no T×T score matrix stored) and sequence parallelism.
- **Inference memory:**
  - weights in the selected dtype;
  - KV cache = layers × cached values per token × context × batch, split by KV head across TP ranks. MLA caches a compressed latent that is replicated across TP ranks.
- **Collectives** use ring-algorithm volumes: all-reduce sends 2(n−1)/n of the tensor per GPU; all-gather, reduce-scatter and all-to-all send (n−1)/n. Rank order (innermost first) is TP, CP, DP, PP; a group that fits inside one node uses NVLink, otherwise the network.
- **Prefill vs decode:** FLOPs = 2 × (parameters used in matrix multiplications) × tokens, plus attention. Bytes = weights read + KV cache read or written. Time = max(FLOPs ÷ peak, bytes ÷ HBM bandwidth).

**Limits.** Times are lower bounds: they assume peak throughput, nominal link bandwidth and no overlap between compute and communication. Activation sizes and the runtime reserve are estimates and are marked "≈". The expert-parallel all-to-all volume assumes every routed token copy is sent separately in BF16, which overstates traffic for node-limited routing.

## Files

| File | Contents |
|---|---|
| `index.html` | page skeleton; loads the scripts below in order |
| `hardware.js` | target system and GPU specification |
| `models.js` | model configs, reasons for inclusion, presets; template families |
| `techniques.js` | deep-dive topics and which models use each |
| `calc.js` | every formula: parameters, memory, transfers, collectives, roofline |
| `app.js` | the model / phase / lens views and controls |
| `technique.html` | deep-dive page template |
| `style.css` | styles |

**Adding a model:**
1. Append an entry to `models.js`, with values from the model's published `config.json` and a `why` stating what it demonstrates.
2. Check that the page's parameter count matches the published total.
3. Give the entry training and inference presets that fit in GPU memory.

**Adding a GPU:** fill the blank slot in `hardware.js`.

## Sources

- Model configs: each model's `config.json` on Hugging Face (linked on the page).
- Llama 3 paper ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)): 405B architecture and training layout.
- DeepSeek-V3 technical report ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)): training layout, optimizer precision and inference deployment.
- NVIDIA GH200 Grace Hopper Superchip datasheet.
- `HPC_spec_v1.md`: measured specification of the target system.
