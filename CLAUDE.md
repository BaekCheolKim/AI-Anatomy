# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI-Flow: an interactive, client-only visualizer that explains how a specific LLM works, from raw text down to GPU execution. It does **not** run any model. Every number shown is arithmetic on published model configs and GPU datasheets. There is no backend, database or auth; the build output is static files.

The user builds it in review-gated phases and reviews each one before the next begins:

1. `types.ts`, `calc.ts` + tests, Llama 3.1 8B data (no UI)
2. Shell: header, model selector, lens tabs, depth control, URL state sync
3. Architecture lens end to end at depths 1–3, with inspector
4. Scenario bar + Memory lens
5. Inference lens, then Training
6. Parallelism + HPC
7. Add DeepSeek-V3 without touching any component. If a component has to change, fix the abstraction, not the component.

Planned stack from Phase 2 on: Vite + React 18 + TypeScript strict, Tailwind, `@xyflow/react` + dagre, Zustand. The URL query string is the source of truth for the whole view (`?model=…&lens=…&node=…&depth=…&b=…&s=…&gpu=…`).

## Commands

Node.js may not be on PATH on this Windows machine. Check `node --version` first.

```sh
npm install
npm test                                  # vitest run (all tests)
npm run typecheck                         # tsc --noEmit; this is the lint gate (no `any`, strict)
npx vitest run src/lib/calc.test.ts       # one file
npx vitest run -t "kvBytesGQA"            # tests whose name matches
npm run test:watch
```

There is no dev server or build script yet (that arrives in Phase 2).

## Architecture

```
render(model, lens, depth, scenario) -> graph + explanation
```

- **Graph structure** is a pure function of `(model, lens)`: shared lens templates (`LensTemplate`, planned under `src/lenses/`) read `ModelSpec.config`.
- **Depth** only filters that graph by `FlowNode.minDepth`, so expanding never navigates away. Edges are authored at their finest depth; the renderer lifts hidden endpoints to the nearest visible ancestor.
- **Every number** is a pure function of `(model, scenario)` and is carried as a `Derived`.

### The one-file-per-model rule

Adding a model must mean adding exactly one file, `src/data/models/<id>.ts`, that default-exports a `ModelSpec`. The file name must equal `id`. `src/data/registry.ts` discovers these files with `import.meta.glob` and sorts them by computed parameter count.

- No code outside `src/data/models/` may mention a model id. Branch on config **features** (`attention.kind`, `ffn.kind`, `position.kind`), never on `model.id`.
- Config variants are discriminated unions (`MHA | GQA | MLA`, `dense | moe`) with exhaustive `switch`es. A model that needs a genuinely new architectural feature is a schema change: extend the union in `src/types.ts` and fix every compile error it causes.
- Structure unique to a single model goes in `ModelSpec.overrides`: `lenses` takes a `LensPatch` (applied remove → patch → add), and `tensors` adds or replaces catalog tensors by id.

### Numbers: `Derived`

`Derived` is a branded type. Only `derive()` in `src/lib/calc.ts` may create one; `src/architecture.test.ts` fails on `as Derived` anywhere else. Each value records its `FormulaId`, the terms substituted into it (terms may themselves be `Derived`, forming a derivation tree the inspector expands), and an `approx` flag that propagates upward.

To add a formula:

1. Add the id to `FormulaId` in `types.ts`.
2. Add a `FORMULAS` entry (name, unit, KaTeX, caveats). The `Record` type forces this.
3. Add a function in `calc.ts` that returns `derive(...)`.
4. Add a test with a hand-computed expected value.

### Tensors

`src/lib/tensors.ts` builds one catalog per model from its config features. Nodes and edges reference tensors by id, so the same KV-cache object appears in every lens.

- **Shape:** a list of dimension expressions (`'S+G'`, `'Nkv*D'`, `'min(pp,mb)'`) in the tiny grammar in `src/lib/expr.ts`. They are resolved through the symbol glossary in `src/lib/dims.ts`. A symbol that doesn't apply to a model (e.g. `Rkv` on a GQA model) resolves to `undefined`, and evaluating it throws.
- **Size:** bytes = numel × `bytesPerParam(dtype)`. `dtype` may be a role (`{ role: 'kv' }`) that follows the scenario's precision controls.
- **Sharding:** `sharding` splits a named shape dimension across a parallel axis. Per-rank size multiplies by ⌈d/w⌉/d, so `tp > kvHeads` correctly replicates heads instead of producing fractions.
- **`explain`:** optional named formula for a tensor's size (e.g. `kvBytesGQA`). Tests assert it equals numel × bytes across the whole scenario sweep.

The HBM budget (`src/lib/budget.ts`) sums the per-rank bytes of catalog tensors that have a `bucket`. It is not a parallel set of formulas; tests cross-check it against `paramsPerRank` in `calc.ts`.

## Modelling conventions (user-approved)

- The FLOP rules (2·N·T forward, 6·N·T training) use **matmul params** `N_mm` = active params minus embedding and position lookup tables, plus the LM head when it is tied. For Llama 3.1 8B that is 7.50B, not 8.03B.
- Memory is shown in binary units (GiB: an "80 GB" H100 is 80 GiB). FLOPs and bandwidth use decimal SI units.
- GPU peak FLOP/s are dense tensor-core figures (datasheets headline 2× with sparsity). Link bandwidth is per direction.
- `Scenario.batch` is the global batch. KV cache is sized for `seqLen + genTokens`, and the decode step is measured at the last position.
- Estimates (activation accounting, runtime overhead, MoE experts touched) set `approx: true` on their `FORMULAS` entry and render with "≈".

## Data sourcing

- Model configs must come from the published `config.json`. Cite the URL in the file header and in `sources`.
- Parameter counts are computed from the config, never copied. Add a test that the computed total matches the published figure.
- If a value can't be verified, ask the user before using it. Where confidence is partial, set `approx: true` on the matching `sources` entry (`field` is a dotted path into the `ModelSpec`) so the UI shows an "approximate" marker.

## Tests

- `src/test/fixtures.ts`: GPT-2 small, Llama 3.1 70B and DeepSeek-V3 configs as unregistered fixtures. They exercise the MHA, learned-position, MLA and MoE paths before those models ship. When one ships as a model file, point the tests at the registry and delete the fixture.
- `src/test/scenarios.ts`: `sweep(m)` generates scenarios across workloads, dtypes and TP/PP/DP/EP layouts for property tests.
- `src/data/models.contract.test.ts`: checks run on every registered model (file name = id, config sanity, every `sources.field` resolves).
- `src/architecture.test.ts`: source-text guards (no model ids outside model files, no `any`, `Derived` only built in `calc.ts`).

TypeScript runs with `strict`, `noUncheckedIndexedAccess` and `verbatimModuleSyntax` (use `import type` for type-only imports).
