# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A static web page that shows how specific models compute, where their tensors reside, and how data moves between devices. The view is organized as **model × phase × lens**:

- **Model:** curated decoder-only LLMs, each included for a stated technique or scale (`why` in models.js), plus template-only families (ViT, latent diffusion, multimodal).
- **Phase:** training, or inference (split into prefill and decode).
- **Lens:**
  - **Data flow:** the stage diagram, with collective-operation markers wherever the parallel layout needs them, plus per-GPU residency in the side panel.
  - **Memory:** what's allocated at load vs materialized per step or per request, transfers, KV-cache growth, prefill vs decode on the roofline.
  - **Parallelism:** the 5 dimensions DP/ZeRO/FSDP, TP+SP, PP, EP and CP, each with what it shards and what collectives it costs.

It runs no model. Open `index.html` directly in a browser (it works from `file://`). There is no build step, package manager, framework or test suite, and the user wants it kept that way. A React/Vite version was deleted for being too complex. Don't add dependencies, build tooling, themes or features the user hasn't asked for, and ask first if something seems to need them.

## Writing rule

Precise technical terminology only: no metaphors, analogies or personification. State operations, shapes, formulas, byte counts and links. This applies to page text and to explanations given to the user.

## Files

Classic scripts loaded in this order by `index.html`; each defines globals used by the next.

- `hardware.js`: `window.GPUS`, with HBM capacity and bandwidth, dense peak FLOP/s per dtype, NVLink / NIC / PCIe per-direction bandwidth, and SRAM. Also `GPUS_PER_NODE`.
- `models.js`: `window.FAMILIES` (template families have an `outline` only) and `window.MODELS`. Per model: config fields, `why`, `source`, optional `trainPrecision`, and `presets.training` / `presets.inference` (parallel degrees + workload + `note`). The note says which values are published and cites the source; everything else is an example.
- `techniques.js`: `window.TECHNIQUES` (id, title, group, `uses(m)` predicate) and `TECHNIQUE_SECTIONS`.
- `calc.js`: every number on the page. `params`, `layout` (shards, DP groups, micro-batches), `validate`, `trainingMemory` / `inferenceMemory` (rows split into `persistent` and `perStep`), `kvGrowth`, `inferenceSteps` (prefill vs decode roofline), `trainingCompute`, `trainingCollectives` / `inferenceCollectives`, `transfers`, plus the `fmt*` formatters.
- `app.js`: UI state (`state` = model, phase, step, lens, stage, scenario `S`), synced to the URL hash. `buildStages(m, S)` returns the data-flow stages; marker stages (`kind: 'coll'`) carry collective rows from calc.js. `renderMemory` and `renderParallel` draw the other lenses.
- `technique.html`: the deep-dive template, `technique.html?id=<technique id>`. It shows the fixed section headings with placeholders, plus "models that use it" computed from `uses(m)`.

## Modelling conventions (in calc.js)

- **Rank order**, innermost first: TP, CP, DP, PP. A group inside one 8-GPU node uses NVLink; otherwise the NIC. Collective volumes use ring algorithms (all-reduce 2(n−1)/n·M; all-gather, reduce-scatter and all-to-all (n−1)/n·M). Times are volume ÷ nominal per-direction bandwidth, with no overlap.
- **Training precision:**
  - BF16 weights, FP32 gradient buffer, FP32 master weights.
  - FP32 AdamW moments, unless the model sets `trainPrecision` (DeepSeek-V3 uses BF16 moments, per its report).
  - ZeRO-1/2/3 shard the optimizer states / + gradients / + weights over the DP×CP group. For experts, the group is the expert-DP group.
- **Experts:** split over EP when ep > 1, otherwise TP-split.
  - The EP all-to-all volume assumes each routed copy is sent separately in BF16 (an overestimate for node-limited routing).
- **Estimates:** activations and runtime reserve are flagged `approx` and rendered with ≈.

## Adding a model

Append to `window.MODELS` using values from the published `config.json`, citing the URL in `source`. Every entry needs a `why` naming the technique or scale it demonstrates; the user removed non-distinctive models. Check that `params(m).total` (and `.active` for MoE) matches the published count. Give it presets that fit in HBM, and mark which preset values are published. Never branch on a model id; extend the config fields and the branches in calc.js and `buildStages` instead.

## Checking a change

Open `index.html`. Go through every model × phase (both inference steps) × lens, step with Prev / Next in the data flow, and check the console for errors. Confirm that presets fit in HBM.
