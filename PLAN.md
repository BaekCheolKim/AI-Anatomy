# Plan

How large AI models are trained, served and run, shown with figures computed from each model's published config. Results are parameter counts, bytes and FLOPs, so they hold on any hardware; the reader sets GPU memory and GPU count. The order of topics follows Andrej Karpathy's video [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (2025). The video is a reference only. It doesn't cover model architectures in detail, ways to run models, or other model families, so those parts are added (marked + below).

## Chapters

Status: all chapters are built. Each chapter has 2–4 plain sentences, figures computed for the selected model (and for all models where a quantity can be compared), a "Formulas and shapes" section, and sources.

### Overview
| Chapter | Figure | Status |
|---|---|---|
| The whole process | Data → pre-training → post-training → RL → serving in one row; a technique × model table, each cell linked to its chapter and source | **built** |

### Pre-training
| Chapter | Figure | Per model | Status |
|---|---|---|---|
| Data | Filter stages from web page to training text, with one sample document after each stage | Published training tokens | **built** |
| Tokenization | Byte-pair encoding trained on the paragraph above the figure, one merge per step | Vocabulary 32,000 to 248,320 | **built** |
| Input and output | A window of token IDs produces one probability per vocabulary entry | Context length, vocabulary | **built** |
| Inside the network | Layer stack drawn to scale; click a layer to follow one token through it, with tensor shapes | Layer pattern, parameter breakdown | **built** |
| + Attention variants | What one position stores per layer: MHA, GQA, MLA, sliding window, linear-attention state | GPT-2, Llama, DeepSeek-V3, gpt-oss, Qwen3.5 | **built** |
| + Mixture of experts | Routing in one layer; total vs per-token parameters of every model; parameters by component; experts split over the reader's GPUs with dispatch bytes | All models | **built** |
| + Images as tokens | Image → patches → vision encoder → visual tokens in the text sequence; tokens per image size | Qwen3-VL, Qwen3.5 | **built** |
| Training loop | Forward → loss → backward → update; bytes per parameter for weights, gradients and optimizer state (AdamW 2 states, Muon 1); FLOPs ≈ 6 × active parameters × tokens | Llama 3.1 405B: 7.3 TB of training state | **built** |
| + Parameters and training tokens | Models placed by parameters and published training tokens, with the ≈ 20 tokens per parameter line (Chinchilla) | Published tokens | **built** |
| + Long context | RoPE base, scaling (YaRN, Llama 3 scaling), context-extension stages; cache size against context | Context lengths | **built** |
| + Splitting across GPUs (optional) | DP, TP, PP, EP, CP: what each splits, which collective, bytes per GPU per step, inside or between nodes | Published layouts of Llama 3.1 405B and DeepSeek-V3 | **built** |
| Sampling from the base model | Temperature and top-p applied to one next-token distribution | — | **built** |

### Post-training
| Chapter | Figure | Per model | Status |
|---|---|---|---|
| Conversations as tokens | One conversation in each model's chat template; which tokens get loss; reasoning formats (gpt-oss harmony channels) | Chat templates from tokenizer_config.json | **built** |
| + Fine-tuning: full or LoRA | Trainable parameters and memory for full fine-tuning vs LoRA at rank r | Llama 2 70B (MLPerf LoRA) | **built** |
| + Distillation | A small model trained on a large model's outputs or logits | Published examples | **built** |
| Context, tools and thinking | Context window split into system, user, tool call, tool result, answer; fixed compute per token (2 × active parameters), so long answers spread computation over tokens; tokens vs letters | FLOPs per token | **built** |

### Reinforcement learning
| Chapter | Figure | Per model | Status |
|---|---|---|---|
| RL with checkable answers | GRPO: G answers per prompt, rewards, advantages normalized within the group, KL to a reference model | DeepSeek-R1 (trained from DeepSeek-V3) | **built** |
| + One RL step on GPUs | Timeline generate → score → reference → train → copy weights, with GPU memory in each phase (generation and training on the same GPUs) | NeMo-RL GRPO recipe for Qwen3.5-397B: 32 prompts × 8 answers | **built** |
| Preference training | Human rankings → reward model → PPO; DPO without a reward model | Llama 2-Chat (PPO), Llama 3.1 (DPO) | **built** |

### + Running a model
| Chapter | Figure | Per model | Status |
|---|---|---|---|
| Prefill and decode | FLOPs and bytes per step, FLOP per byte | Absorbed MLA, MTP draft tokens | **built** |
| KV cache and recurrent state | Bytes per token and per sequence, growth with context against GPU memory | Qwen3.5: 30,720 B per token + 189 MB per sequence (FP32 state); Qwen3-VL: 192,512 B per token | **built** |
| Precision and quantization | BF16, FP8, INT8, INT4 (AWQ, GPTQ), MXFP4, NVFP4, GGUF: bytes per weight including scales; which tensors stay in BF16 | Published formats | **built** |
| Will it fit? | Weights + KV cache at context × concurrent sequences + workspace, against GPU memory × count; minimum GPUs per model and precision | All models | **built** |
| Inference engines | Mechanisms: paged KV cache, continuous batching, prefix caching, chunked prefill, speculative decoding, CUDA graphs, parallel serving; engine settings (max context, memory fraction, KV dtype, parallel size) mapped to the memory figure; vLLM, SGLang, TensorRT-LLM, llama.cpp | — | **built** |
| Serving many requests | Batch size vs time to first token, time per output token and tokens/s; prefill/decode disaggregation | gpt-oss-120b on one GPU, Qwen3-VL FP8 on four | **built** |
| From request to response | HTTP API → chat template → tokens → scheduler → prefill → decode → detokenize → stream; structured output by masking logits | — | **built** |
| Agents | Repeated calls with tool results appended; context and KV growth; prefix-cache reuse | — | **built** |

### + Other model families
| Chapter | Figure | Model | Status |
|---|---|---|---|
| Embedding models and retrieval | Text → one vector; retrieval of documents into an LLM's context | Qwen3-Embedding-8B (checkpoint count matched) | **built** |
| Image generation | Diffusion / flow matching: training target, N denoising steps through the transformer, VAE decode; memory and FLOPs per image | FLUX.1 [schnell] (configs gated; the architecture reproduces the checkpoint count) | **built** |
| Speech recognition | Audio → log-mel spectrogram → encoder → decoder with cross-attention → text | Whisper large-v3 (checkpoint count matched) | **built** |

## Models

| Model | Shows |
|---|---|
| GPT-2 1.5B | 2019 design: MHA, learned positions, LayerNorm, GELU, tied embeddings |
| Llama 3.1 8B | Dense GQA; MLPerf Training benchmark |
| Llama 2 70B | LoRA fine-tuning (MLPerf) |
| Llama 3.1 405B | Largest dense model; published 4D-parallel layout |
| DeepSeek-V3 | MLA, 256 experts + 1 shared, MTP, FP8 training; base of DeepSeek-R1 |
| Qwen3-VL-235B-A22B | Images as tokens; 128 experts |
| Qwen3.5-397B-A17B | 45 Gated DeltaNet + 15 gated-attention layers, 512 experts, MTP, image input |
| gpt-oss-120b | MXFP4 experts, sliding-window/full attention, attention sinks, harmony format |

Optional: DeepSeek-V3.2 (sparse attention), Kimi K2 (Muon optimizer). Every model's computed checkpoint parameter count equals the Hugging Face safetensors count (see CLAUDE.md, Adding a model).

## Measured values next to computed ones

In `measured.js`, shown as a table in the chapter each belongs to (KV cache, Precision, Inference engines, Serving, Splitting across GPUs). The computed column comes from configs and holds on any hardware; the measured column states its conditions. No job IDs, node names, host names or internal paths: this repository is public.

| Quantity | Computed | Measured | Conditions |
|---|---|---|---|
| KV cache per token per GPU, Qwen3-VL-235B-A22B, TP 4, BF16 cache | 48,128 B | 48,144 B (22.22 GiB / 495,568 tokens) | vLLM 0.32.0, FP8 weights, 4 × GH200 96 GB |
| FP8 checkpoint size, same model | 238 GB | 238 GB (222 GiB) on disk | published FP8 release |
| Weights per GPU, same setup | 56.11 GiB (vision encoder copied to every GPU) | 56.47 GiB | vLLM 0.32.0, TP 4 |
| Model loading | a quarter of the checkpoint per GPU | 167.8 s; each process read all 238 GB | TP 4, one node, shared file system |
| Output tokens/s, 64 concurrent text requests | — | 1,394.65 | vLLM 0.32.0, FP8, TP 4 |
| 512 MiB all-reduce, 2 nodes × 4 GPUs | 134,217,728 B per GPU between nodes | 3.4 ms | 4 × 400 Gb/s NICs per node |

## Formula fixes

All applied: expert-parallel bytes per GPU (`inferenceStep` divides instance totals over all GPUs, so a GPU reads only the weights it holds: DeepSeek-V3 decode 23.8 GB per GPU, against 104 GiB before); absorbed MLA FLOPs in decode (`attnFlopsPerToken`); collective counts under full recomputation and ZeRO-0 (`trainingCollectiveRows`); two-level collective bytes (`collectiveBytes`); unknown model ids in links fall back to the default.

## Order

1. **Foundations** (done): CLAUDE.md, README, this plan; hardware reduced to GPU memory and count; parameter counts checked against checkpoints; Mixture of experts chapter as the style prototype (approved).
2. **Refactor** (done): weights by tensor class and dtype, KV-cache dtype, layer types and linear-attention state, FLOPs by layer type, training state and activations, inference steps.
3. **Models** (done): `layouts.training` for every model (published for Llama 3.1 405B and DeepSeek-V3), chat formats, published post-training, RoPE scaling; three models of other families in `OTHER_MODELS`.
4. **Chapters** (done): 31 chapters, from the overview to speech recognition.
5. **Measured values** (done): `measured.js`, shown in their chapters.

Possible next steps: DeepSeek-V3.2 (sparse attention) and Kimi K2 (Muon optimizer) as optional models; more measured values from new runs.
