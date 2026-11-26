# AI-Anatomy

A static web page that shows, with figures, how large AI models are trained, served and run, and the techniques each model uses. Every number is computed from the model's published configuration, so it holds on any hardware. The only hardware inputs are GPU memory and GPU count, which the reader sets. The page runs no model.

## Opening it

Open `index.html` in a browser. There is nothing to install, build or serve; it also works from `file://`.

## What it shows

Chapters follow the order of Andrej Karpathy's video [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (2025): pre-training, post-training, reinforcement learning. The page extends that order with model architectures (attention variants, mixture of experts, images as tokens), running models (precision, memory, inference engines, serving) and other model families (embeddings, image generation, speech recognition). Text and figures are original. [PLAN.md](PLAN.md) lists every chapter.

Models, each included for a technique or scale:

| Model | Shows |
|---|---|
| GPT-2 1.5B | The 2019 design: multi-head attention, learned positions, LayerNorm, GELU, tied embeddings |
| Llama 3.1 8B | Dense model with grouped-query attention |
| Llama 2 70B | LoRA fine-tuning |
| Llama 3.1 405B | Largest dense open-weight model; published 4D-parallel training layout |
| DeepSeek-V3 | Multi-head latent attention, 256 small experts + 1 shared, multi-token prediction, FP8 training |
| Qwen3-VL-235B-A22B | Vision encoder + MoE language model |
| Qwen3.5-397B-A17B | Gated DeltaNet + gated attention layers, 512 experts, multi-token prediction |
| gpt-oss-120b | 4-bit MXFP4 experts, sliding-window and full attention, attention sinks |

Other model families, each in its own chapter: Qwen3-Embedding-8B (retrieval), FLUX.1 [schnell] (image generation), Whisper large-v3 (speech recognition).

## How the numbers are computed

All formulas are in `calc.js`.

- **Parameters** are counted tensor by tensor from each config. For every model, the eleven of them, the count equals the checkpoint's parameter count on Hugging Face; for example, Qwen3.5-397B-A17B gives 403,397,928,944, which includes the vision encoder and multi-token-prediction module.
- **Bytes** are in decimal units (1 GB = 10⁹ bytes): parameters × bytes per parameter of the chosen precision.
- **Active parameters** are those used for one text token: unselected experts and the vision encoder are left out.
- **Collective traffic** is computed per GPU, inside a node and between nodes, for a group of n GPUs with r of them per node.
- **Illustrative values** that can't be computed from a config, such as which experts an example token selects, are labelled as such on the page.
- **Measured values** from real runs appear next to the computed ones, with their conditions (`measured.js`). Times appear only there, or when a reader enters GPU speeds.

## Files

| File | Contents |
|---|---|
| `index.html` | Page skeleton; loads the scripts below in order |
| `models.js` | Model configs, reasons for inclusion, checkpoint and published counts |
| `calc.js` | Every formula |
| `figures.js` | SVG and other helpers shared by chapters |
| `measured.js` | Measured values with their conditions |
| `chapters/*.js` | One file per built chapter |
| `app.js` | Top bar, chapter list, URL state |
| `style.css` | Styles |
| `PLAN.md` | Chapters, models, measured values, order of work |
