// Chapter: The whole process.
// Figures: (1) the path from data to a served model, each stage linked to its chapters; (2) which
// techniques each model uses, derived from the model fields, each linked to the chapter that shows it.
(function () {
  const STAGES = [
    ['Data', 'web text, filtered, counted in tokens', ['data', 'tokenization']],
    ['Pre-training', 'next-token prediction on trillions of tokens', ['io', 'network', 'attention', 'moe', 'images', 'training', 'scaling', 'long-context', 'parallel', 'sampling']],
    ['Post-training', 'conversations, fine-tuning, distillation', ['chat', 'lora', 'distillation', 'context']],
    ['Reinforcement learning', 'rewards from rule-based answer checks or from reward models trained on human preferences', ['rl', 'rl-gpus', 'preferences']],
    ['Running', 'prefill, decode, caches, engines, serving', ['prefill-decode', 'kv-cache', 'precision', 'fit', 'engines', 'serving', 'request', 'agents']],
  ];

  // [technique, chapter id, test on a model]
  const TECH = [
    ['Multi-head attention', 'attention', (m) => m.attention.kind === 'MHA'],
    ['Grouped-query attention', 'attention', (m) => m.attention.kind === 'GQA'],
    ['Multi-head latent attention', 'attention', (m) => m.attention.kind === 'MLA'],
    ['Sliding-window layers', 'attention', (m) => layerTypes(m).includes('sliding')],
    ['Linear-attention layers', 'attention', (m) => layerTypes(m).includes('linear')],
    ['Attention sinks', 'attention', (m) => Boolean(m.attention.sinks)],
    ['Gated attention output', 'network', (m) => Boolean(m.attention.outputGate)],
    ['QK-norm', 'network', (m) => Boolean(m.attention.qkNorm)],
    ['Mixture of experts', 'moe', (m) => isMoE(m)],
    ['Shared expert', 'moe', (m) => isMoE(m) && m.ffn.shared > 0],
    ['Multi-token prediction', 'prefill-decode', (m) => Boolean(m.mtp)],
    ['Vision encoder', 'images', (m) => Boolean(m.vision)],
    ['Learned positions', 'long-context', (m) => m.position === 'learned'],
    ['Rotary positions with scaling', 'long-context', (m) => Boolean(m.ropeScaling)],
    ['Below-BF16 release', 'precision', (m) => Object.values(releaseDtypes(m)).some((d) => DTYPE_BYTES[d] < 2)],
    ['Chat format', 'chat', (m) => Boolean(m.chat)],
    ['DPO', 'preferences', (m) => Boolean(m.postTraining?.methods?.includes('dpo'))],
    ['RLHF with PPO', 'preferences', (m) => Boolean(m.postTraining?.methods?.includes('ppo'))],
    ['RL (GRPO or other)', 'rl', (m) => Boolean(m.postTraining?.methods?.some((x) => x === 'grpo' || x === 'rl'))],
    ['Published training layout', 'parallel', (m) => !m.layouts?.training?.example],
  ];

  const title = (id) => CHAPTERS_FLAT.find((c) => c.id === id).title;
  const link = (id, text) => `<a href="#ch=${id}" data-ch="${id}">${esc(text ?? title(id))}</a>`;

  function figStages(W) {
    const narrow = W < 560;
    return `<div class="stages-row${narrow ? ' narrow' : ''}">${STAGES.map(([name, what, ids], i) => `<div class="stage"><p class="part">${i + 1} ${esc(name)}</p><p class="f">${esc(what)}</p><ul>${ids.map((id) => `<li>${link(id)}</li>`).join('')}</ul></div>`).join('')}</div>
      <p class="caption">Each stage produces the input of the next: filtered text, a base model, an assistant, an assistant further trained against rewards (a reasoning model when the rewards check final answers), and a model served to users. Other model families (embeddings, image generation, speech) also go through data preparation, training and serving, with different inputs, objectives and training stages: ${link('embeddings', 'embeddings')}, ${link('diffusion', 'image generation')}, ${link('speech', 'speech recognition')}.</p>`;
  }

  function figMatrix(m) {
    const models = window.MODELS;
    const head = `<tr><th>technique</th>${models.map((x) => `<th class="c${x.id === m.id ? ' on' : ''}"><a href="#" data-model="${x.id}">${esc(x.short)}</a></th>`).join('')}</tr>`;
    const rows = TECH.map(([name, ch, test]) => `<tr><td>${link(ch, name)}</td>${models.map((x) => `<td class="c${x.id === m.id ? ' on' : ''}">${test(x) ? '●' : ''}</td>`).join('')}</tr>`).join('');
    return `<table class="data matrix"><thead>${head}</thead><tbody>${rows}</tbody></table>
      <p class="caption">Derived from each model's config and published reports (models.js); a technique name links to the chapter that shows it, a model name selects the model. Post-training methods are as published; an empty cell can also mean not published.</p>`;
  }

  function render(ctx) {
    const W = ctx.width;
    return `
      <p class="lead">A large language model is made in stages. Web text is filtered and converted to tokens; pre-training trains a network to predict the next token; post-training on conversations turns it into an assistant; reinforcement learning improves its answers where they can be scored; and an inference engine serves it to many users. This page follows those stages with figures computed from the published configurations of eight models. Computed numbers hold on any hardware, and published or measured figures carry their sources; GPU memory and GPU count, set at the top, are the only hardware inputs.</p>
      <h2>From data to a served model</h2>${figStages(W)}
      <h2>Techniques of each model</h2>${figMatrix(ctx.m)}
      <p class="sources">The order of topics follows Andrej Karpathy's video <a href="https://www.youtube.com/watch?v=7xTGNNLPyMI" target="_blank" rel="noreferrer">Deep Dive into LLMs like ChatGPT</a> (2025). Model configurations and reports are cited in each chapter.</p>`;
  }

  (window.CHAPTERS = window.CHAPTERS || {}).overview = { render };
})();
