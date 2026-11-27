# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A static web page that shows, with figures, how large AI models are trained, served and run, and the techniques each model uses. Parameter counts, bytes and FLOPs are computed from the selected model's published config; published figures and measured values carry their sources and conditions.

- **Order:** chapters follow Andrej Karpathy's video "Deep Dive into LLMs like ChatGPT" (pre-training, post-training, reinforcement learning). The video is a reference only. The page adds what it doesn't cover: model architectures, running models, and other model families. PLAN.md lists every chapter and its status.
- **Hardware-independent:** results are parameter counts, bytes and FLOPs. The reader sets GPU memory and GPU count in the top bar. These are the only hardware inputs, apart from GPUs per node where parallel groups are placed. Times appear only as measured values with their conditions, or when the reader enters optional GPU speeds (peak FLOP/s, memory bandwidth, link bandwidth) in a chapter.
- **Model selector:** one model at a time in the top bar. Figures compare all models where a quantity is comparable.

It runs no model. Open `index.html` directly in a browser (it works from `file://`). There is no build step, package manager, framework or test suite, and the user wants it kept that way. A React/Vite version was deleted for being too complex. Don't add dependencies, build tooling or themes, and ask first if something seems to need them.

## Writing rule

Precise technical terminology only: no metaphors, analogies or personification. State operations, shapes, formulas, byte counts and links. Where the video uses an analogy, state the mechanism instead. This applies to page text and to explanations given to the user.

## Style rule

The user rejected the earlier look (bordered and rounded boxes, blue accents) as "too Claude-themed".
- White page, black and gray text, hairlines. No cards, rounded boxes, shadows, or accent colors in the interface. Controls are native inputs and text links; the selected item is underlined.
- Color appears only in figures and encodes the kind of data, with the same mapping everywhere (CSS variables in style.css): weights green, activations orange, KV cache reddish purple, gradients and optimizer state vermillion, communication yellow. Lighter and darker shades of one hue encode states of the same data, for example stored but not selected.
- Figures are SVG drawn at the chapter column's width (`ctx.width`), so their text keeps its size. Check a 390 px viewport; there is no horizontal page scroll.

## Files

Classic scripts loaded in this order by `index.html`; each defines globals used by the next.

- `models.js`: `window.MODELS` (the eight models of the model bar) and `window.OTHER_MODELS` (embedding, diffusion and speech models used in their own chapters). Per model: config fields, `why`, `source`, `checkpoint`, `published`, `release` (dtypes of the published weights by tensor class), `layouts.training`, `chat`, `postTraining`, and optional `training`, `layerTypes`, `linearAttention`, `mtp`, `vision`, `ropeScaling`, `longContext`, `chatBase` and `chatThink` (chat format of a base checkpoint's post-trained model; a generation prompt that opens `<think>`), `layouts.rl` (a published RL recipe), `gated` and `sourceNote` (config needs a login; where the values come from). The header comment defines the fields.
- `calc.js`: every number. Parameters (`params`, with `total`, `checkpoint`, `active`, `matmul`); weights by class and dtype (`weightClasses`, `releaseDtypes`, `weightBytes`); MoE (`expertsTouched`, `dispatchBytes`); caches (`cacheValues`, `cacheBytes`, `kvBytesPerToken`, `kvBytesPerTokenPerGPU`, `cacheBytesPerGPU`); FLOPs (`attnFlopsPerToken`, `forwardFlopsPerToken`, `trainingFlops`); training (`trainBytesPerParam`, `savedValuesPerToken`, `trainingLayout`, `validateLayout`, `trainingMemoryPerGPU`, `trainingCollectiveRows`); inference (`inferenceStep`, `stepTime`, `servingLayout`: TP inside a node, PP across nodes, TP dividing the heads); collectives (`groupPlacement`, `collectiveBytes`); `fmt*` formatters.
- `figures.js`: `esc`, `figure` (SVG), `logScale`, a seeded random generator for illustrative examples, `dtypeSelect`, `configLink` (a model's config.json link, marked when gated), the GPUs-per-node setting shared by the serving chapters (`SERVE`, `gpnSelect`, `bindGpn`), and the optional GPU-speed inputs (`hwInputs`, `bindHw`, `hwPeak`, `hwBw`).
- `measured.js`: `window.MEASURED`, values from the user's runs with their conditions, and `measuredTable(chapter, m)`.
- `chapters/<id>.js`: one file per chapter. It registers `window.CHAPTERS[id] = { render(ctx) → HTML, mount(root, ctx, rerender) }`, where `ctx = { m, mem (bytes), gpus, width }`. Its `<script>` tag goes in index.html before app.js.
- `app.js`: `OUTLINE` (all chapters in order), the top bar, the chapter list, URL-hash state (`ch`, `model`, `mem`, `gpus`; unknown values fall back to defaults), ← → between chapters.
- `PLAN.md`: chapters, models, measured values, formula fixes, order of work.

## Modelling conventions (in calc.js)

- **Bytes:** decimal units (1 GB = 10⁹ B), so parameters × bytes per parameter reads directly in GB. Measured values keep their original unit next to the conversion.
- **Parameters:**
  - `total` = language model + vision encoder, excluding multi-token-prediction modules; publishers' headline counts use this.
  - `checkpoint` = `total` + MTP (+ MTP copies of the embedding and output matrices when stored).
  - `active` = parameters used for one text token: everything except unselected routed experts and the vision encoder, with the embedding table counted in full, as DeepSeek and Qwen count it. OpenAI's gpt-oss count leaves the embedding table out; `published.convention` says so.
  - `matmul` = `active` − embedding table: the N in 2·N FLOPs per token.
- **Layer types:** `full`, `sliding` (K/V for at most `window` positions), `linear` (Gated DeltaNet: fixed state of valueHeads × keyDim × valueDim per layer, in `stateBytes`; Qwen3.5 keeps it in FP32 per `mamba_ssm_dtype`).
- **Collectives:** rank order is TP, CP, DP, PP, innermost first. A group of n ranks with r per node spans N = n/r nodes. All-gather and reduce-scatter send (r−1)/r·M inside the node and (N−1)/N·M/r across nodes per GPU; all-reduce twice that; all-to-all sends (n−r)/n·M across nodes. A measured 512 MiB all-reduce on 2 nodes × 4 GPUs (4 × 400 Gb/s NICs per node) took 3.4 ms, 79% of the bandwidth this formula implies; the old per-GPU-NIC formula was 5.5× slower than measured.
- **Examples that aren't computed** (which experts an example token selects, sample text) are labelled illustrative on the page.

## Adding a model

1. Append to `window.MODELS` using values from the published `config.json`, and cite its URL in `source`.
2. Give it a `why` naming the technique or scale it shows; the user removed non-distinctive models.
3. Set `checkpoint.params` to the count from `https://huggingface.co/api/models/<repo>` (its `safetensors.total`). `params(m).checkpoint` must equal it, minus non-parameter buffers stored in the checkpoint (`checkpoint.buffers`, e.g. GPT-2's causal masks). If it doesn't, a structural field is missing.
4. Record the authors' headline counts and their convention in `published`, and the published weights' dtypes in `release`.

Never branch on a model id; extend the config fields and calc.js instead.

## Measured values

Measured numbers come from the user's runs and must state their conditions: software versions, GPU type and count, precision. This repository is public. Never include job IDs, node names, host names, IP addresses, account names, internal paths, or allocation and quota figures.

## Checking a change

Open `index.html`. Go through every chapter × every model, use each control, and check the console for errors and for NaN or undefined in the text. Check a 390 px wide viewport (no horizontal page scroll; wide tables scroll inside themselves). Confirm parameter counts against `checkpoint.params`. A model with a short context (GPT-2, 1,024 tokens) must still give every length menu at least one option.
