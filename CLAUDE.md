# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single static web page that explains how a transformer LLM turns a prompt into the next token, with the whole pipeline visible at once and the connections between concepts made explicit. It is an illustration: it runs no model.

It is three files, opened directly in a browser (double-click `index.html`, which works from `file://`):

- `index.html`: the diagram markup. Every clickable concept is an element with `data-id="<stage>"`.
- `app.js`: all content and behaviour (a plain classic script, not a module).
- `style.css`: plain CSS.

There is no build step, package manager, framework or test suite, and the user wants it kept that way. A previous React/Vite/Tailwind version was deliberately deleted for being too complex. Don't add dependencies, build tooling, themes or other features the user hasn't asked for. Ask first if something seems to need them.

## How app.js is organised

- `MODEL`: config values from Llama 3.1 8B's published `config.json` (layers, hidden, heads, kvHeads, headDim, ffn, vocab). Derived numbers (parameter counts, percentages, KV bytes per token) are computed from `MODEL` in the "numbers derived from the config" block, never typed in by hand.
- `TOKENS`, `IDS`, `NEXT`: the running example. These values are **illustrative**, and the page says so. Keep that labelling whenever you add example values.
- `STAGES`: one entry per `data-id`, holding `title`, `what`, optional `why`, `shapes` (label/value rows), optional `note`, and `links` (`[targetId, phrase]`), which are the "How it connects" relations. Linked boxes are highlighted when a stage is selected.
- `ORDER`: the Prev / Next walk-through order. A new stage needs a box in `index.html`, an entry in `STAGES` and a place in `ORDER`.
- The small visuals (attention grid, SiLU curve, probability bars) are built as inline SVG/HTML strings by `attentionGrid()`, `siluChart()` and `probBars()`.

## Checking a change

Open `index.html` in a browser, click through the boxes and the Prev / Next stepper, and check the console for errors.
