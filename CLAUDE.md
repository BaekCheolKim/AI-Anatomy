# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single static web page that explains how a transformer LLM turns a prompt into next-token probabilities. The whole pipeline is visible at once, and the connections between stages are explicit. It illustrates the computation; it runs no model.

It is four plain files, opened directly in a browser (double-click `index.html`; it works from `file://`):

- `index.html`: the page skeleton. Two classic `<script>` tags load `models.js` then `app.js`.
- `models.js`: `window.MODELS`, one config object per model.
- `app.js`: the template that builds the diagram and all text from the selected model.
- `style.css`: plain CSS.

There is no build step, package manager, framework or test suite, and the user wants it kept that way. A previous React/Vite/Tailwind version was deleted for being too complex. Don't add dependencies, build tooling, themes or features the user hasn't asked for, and ask first if something seems to need them.

## Writing rule

Use precise technical terminology only: no metaphors, analogies or personification. For example, write "S = Q·Kᵀ/√d_h; causal mask; A = softmax(S)", not "tokens look at each other". This applies to all page text and to explanations given to the user.

## Adding a model

Append one entry to `window.MODELS` in `models.js`, with every value taken from the model's published `config.json` and the URL in `source`. Field meanings are documented at the top of that file:

- `attention`: `kind` is `MHA`, `GQA` or `MLA`; `bias` is `none`, `qkv` or `all`.
- `ffn`: `kind` is `dense` or `moe`; `act` is `silu` or `gelu`; `gated` marks a SwiGLU/GeGLU FFN.
- `position` is `learned` or `rope`; `norm` is `LayerNorm` or `RMSNorm`; `tied` marks a shared embedding/LM-head matrix.

Then open the page and check that `params(m).total` (and `.active` for MoE) matches the published parameter count. `params()` counts every matrix from the config. If a new model needs an architecture variant these fields can't express, extend the schema and the branches in `buildStages()`. Never branch on the model id.

## How app.js is organised

- `TOKENS`, `IDS`, `NEXT`: the running example ("The cat sat on the"). These values are **illustrative**, as are the attention grid and the expert selection, and the page labels them so. Keep that labelling for any new example values.
- `params(m)`, `kvValuesPerToken(m)`: all derived numbers come from these functions, never from hand-typed figures.
- `buildStages(m)` returns `{ input, residual, layer, output }`. Each stage has `id`, `title`, optional `tag` / `viz` (HTML) / `children` (nested boxes; `row: true` with `pre` / `post` lays them out horizontally), plus `what`, optional `why`, `shapes` (label / value rows), optional `note`, and `links` (`[targetId, relation]`, shown as "Connections").
- `render(m)` writes the diagram and the model line. `order` (Prev / Next) is the depth-first order of the stages. `select(id)` marks the current box (solid blue, CURRENT badge), the next box (dashed amber, NEXT badge) and related boxes, fills the side panel, and writes `#model=…&stage=…` to the URL hash.
- The small visuals are `attentionGrid()`, `activationChart(m)`, `routerGrid(m)` (which respects group-limited routing) and `probBars()`.

## Planned next

A hardware view for each stage (where tensors live, HBM traffic, and parallelism such as TP/PP/EP sharding and collectives) is requested but not built yet. Add it as an extra per-stage field rendered in the side panel.

## Checking a change

Open `index.html`, switch between all models, click boxes and step with Prev / Next, and check the console for errors.
