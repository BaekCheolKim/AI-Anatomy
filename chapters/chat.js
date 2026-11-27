// Chapter: Conversations as tokens.
// Figure: one example conversation written in the selected model's chat format, with control tokens
// marked and the tokens that receive loss in supervised fine-tuning underlined.
(function () {
  const SYS = 'You are a helpful assistant.', USER = 'How many legs does a spider have?';
  const THINK = 'Spiders are arachnids. Arachnids have four pairs of legs, so 4 × 2 = 8.', ANSWER = 'A spider has eight legs.';
  let thinking = true;

  // Each format: [text, kind] parts. kind 's' control token, 'm' masked text, 't' trained text, 'ts' trained control token.
  // Text the server writes before generation (the assistant header and, for Qwen3.5, the opening <think>) is masked;
  // prompt(m) names it, as HTML.
  const FORMATS = {
    llama3: {
      name: 'Llama 3 chat format', src: 'https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/', prompt: () => 'everything up to and including the assistant header',
      parts: () => [
        ['<|begin_of_text|>', 's'], ['<|start_header_id|>', 's'], ['system', 'm'], ['<|end_header_id|>', 's'],
        [`\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n${SYS}`, 'm'], ['<|eot_id|>', 's'],
        ['<|start_header_id|>', 's'], ['user', 'm'], ['<|end_header_id|>', 's'], [`\n\n${USER}`, 'm'], ['<|eot_id|>', 's'],
        ['<|start_header_id|>', 's'], ['assistant', 'm'], ['<|end_header_id|>', 's'], ['\n\n', 'm'], [ANSWER, 't'], ['<|eot_id|>', 'ts']],
    },
    llama2: {
      name: 'Llama 2 chat format', src: 'https://github.com/meta-llama/llama/blob/main/llama/generation.py', prompt: () => 'everything up to and including <code>[/INST]</code>',
      parts: () => [['<s>', 's'], ['[INST] <<SYS>>\n', 'm'], [SYS, 'm'], ['\n<</SYS>>\n\n', 'm'], [USER, 'm'], [' [/INST] ', 'm'], [ANSWER, 't'], [' ', 't'], ['</s>', 'ts']],
    },
    deepseek: {
      name: 'DeepSeek-V3 chat format', src: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json', prompt: () => 'everything up to and including <code>&lt;｜Assistant｜&gt;</code>',
      parts: () => [['<｜begin▁of▁sentence｜>', 's'], [SYS, 'm'], ['<｜User｜>', 's'], [USER, 'm'], ['<｜Assistant｜>', 's'], [ANSWER, 't'], ['<｜end▁of▁sentence｜>', 'ts']],
    },
    chatml: {
      name: 'ChatML, as used by Qwen', src: 'https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/chat_template.jinja',
      prompt: (m) => (m.chatThink === 'prompt' ? 'the assistant header and the opening <code>&lt;think&gt;</code> line (or, with thinking off, an empty <code>&lt;think&gt;&lt;/think&gt;</code> block)' : 'everything up to and including the assistant header'),
      parts: (m, th) => [
        ['<|im_start|>', 's'], ['system\n', 'm'], [SYS, 'm'], ['<|im_end|>', 's'], ['\n', 'm'],
        ['<|im_start|>', 's'], ['user\n', 'm'], [USER, 'm'], ['<|im_end|>', 's'], ['\n', 'm'],
        ['<|im_start|>', 's'], ['assistant\n', 'm'],
        ...(m.chatThink === 'prompt' ? (th ? [['<think>', 's'], ['\n', 'm'], [`${THINK}\n`, 't'], ['</think>', 'ts'], ['\n\n', 't']] : [['<think>', 's'], ['\n\n', 'm'], ['</think>', 's'], ['\n\n', 'm']]) : []),
        [ANSWER, 't'], ['<|im_end|>', 'ts']],
    },
    harmony: {
      name: 'harmony format (gpt-oss)', src: 'https://github.com/openai/harmony/blob/main/docs/format.md', prompt: () => '<code>&lt;|start|&gt;assistant</code>',
      parts: () => [
        ['<|start|>', 's'], ['system', 'm'], ['<|message|>', 's'],
        ['You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-05\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.', 'm'], ['<|end|>', 's'],
        ['<|start|>', 's'], ['developer', 'm'], ['<|message|>', 's'], [`# Instructions\n\n${SYS}`, 'm'], ['<|end|>', 's'],
        ['<|start|>', 's'], ['user', 'm'], ['<|message|>', 's'], [USER, 'm'], ['<|end|>', 's'],
        ['<|start|>', 's'], ['assistant', 'm'], ['<|channel|>', 'ts'], ['analysis', 't'], ['<|message|>', 'ts'], [THINK, 't'], ['<|end|>', 'ts'],
        ['<|start|>', 'ts'], ['assistant', 't'], ['<|channel|>', 'ts'], ['final', 't'], ['<|message|>', 'ts'], [ANSWER, 't'], ['<|return|>', 'ts']],
    },
  };

  function view(m) {
    const f = FORMATS[m.chat], toggle = m.chat === 'chatml' && m.chatThink === 'prompt', parts = f.parts(m, thinking);
    const html = parts.map(([t, k]) => `<span class="cv-${k}">${esc(t).replace(/\n/g, '↵\n')}</span>`).join('');
    const trained = parts.filter(([, k]) => k === 't' || k === 'ts').reduce((n, [t]) => n + t.length, 0);
    const all = parts.reduce((n, [t]) => n + t.length, 0);
    return `${toggle ? `<p class="controls"><label><input type="checkbox" id="cv-think"${thinking ? ' checked' : ''}> thinking on (reasoning section before the answer)</label></p>` : ''}
      <pre class="doc conv">${html}</pre>
      <p class="caption">Bold: control tokens, each a single vocabulary entry added to the tokenizer; byte-pair merges never produce them from text (DeepSeek-V3's <code>&lt;｜User｜&gt;</code> and <code>&lt;｜Assistant｜&gt;</code> and Qwen's <code>&lt;think&gt;</code> and <code>&lt;/think&gt;</code> are added tokens not flagged as special). Underlined: the tokens that receive loss in supervised fine-tuning (${fmtPct(trained, all)} of the characters here); system, developer and user tokens are part of the input but are not predicted. At inference the server writes ${f.prompt(m)}, and the model generates the rest until its end token.${m.chat === 'harmony' ? ' The dates in the system message are examples.' : ''}</p>`;
  }

  function render(ctx) {
    const m = ctx.m;
    const lead = `<p class="lead">Pre-training produces a model that continues documents. Post-training turns it into an assistant by continuing the same next-token training on conversations: example dialogues written by people or generated by other models, each a system message, user turns and assistant replies. A conversation is written as one token sequence with control tokens that mark where each message starts and ends and who wrote it. The model learns to produce the assistant parts, including the token that ends its turn.</p>`;
    if (!m.chat) {
      const others = window.MODELS.filter((x) => x.chat).map((x) => `<a href="#" data-model="${x.id}">${esc(x.short)}</a>`).join(', ');
      return `${lead}<p><b>${esc(m.name)}</b> was released as a base model without a chat format: it continues whatever text it is given. Select a post-trained model to see its format: ${others}.</p>`;
    }
    const f = FORMATS[m.chat];
    const base = m.chatBase ? ` (used by ${esc(m.chatBase)}; ${esc(m.name)} itself is a base model)` : '';
    return `${lead}
      <p><b>${esc(m.name)}</b>: ${esc(f.name)}${base}. The template is a Jinja program distributed with the tokenizer files (chat_template in tokenizer_config.json, or a separate chat_template.jinja) and applied by the inference server.</p>
      <h2>One conversation, written as tokens</h2>${view(m)}
      <h2>What the format carries</h2>
      <ul class="formulas">
        <li><b>Roles.</b> System, user and assistant messages are delimited by control tokens. A literal string such as <code>&lt;|im_end|&gt;</code> inside a message is encoded as that control token unless the server tokenizes message text with special-token parsing disabled (Hugging Face split_special_tokens; tiktoken rejects such text by default). Llama 2's <code>[INST]</code> and <code>&lt;&lt;SYS&gt;&gt;</code> markers are ordinary text.</li>
        <li><b>Turn ends.</b> The assistant's end token is trained like any other token; generation stops when the model produces it.</li>
        <li><b>Reasoning.</b> Models trained to reason write a separate section before the answer: Qwen inside <code>&lt;think&gt;</code> tags, gpt-oss in an analysis-channel message. In multi-turn conversations, earlier reasoning sections are dropped from the context (harmony; the Qwen3.5 template keeps reasoning only after the last user message).</li>
        <li><b>Tool calls</b> are text in a fixed syntax: JSON after the assistant header (Llama 3.1 custom tools), <code>&lt;tool_call&gt;</code> blocks (Qwen), <code>&lt;｜tool▁calls▁begin｜&gt;</code>…<code>&lt;｜tool▁calls▁end｜&gt;</code> (DeepSeek-V3), or a separate commentary-channel message addressed <code>to=functions.NAME</code> and ending with <code>&lt;|call|&gt;</code> (gpt-oss). The tool's output returns as another message (Context, tools and thinking).</li>
      </ul>
      <details class="more"><summary>Loss</summary><ul class="formulas">
        <li>Supervised fine-tuning loss: L = −(1/|A|) Σ<sub>t ∈ A</sub> ln p(x<sub>t</sub> | x<sub>&lt;t</sub>), where A is the set of positions inside assistant messages. The other positions are in the input, but their predictions are not trained.</li>
        <li>The data is far smaller than pre-training data, but the same next-token objective applies; InstructGPT (2022) trained on about 13,000 prompts with demonstrations written by labelers.</li>
      </ul></details>
      <p class="sources">Sources: <a href="${esc(f.src)}" target="_blank" rel="noreferrer">${esc(f.name)}</a>${m.chat === 'harmony' ? ' · <a href="https://huggingface.co/openai/gpt-oss-120b/blob/main/chat_template.jinja" target="_blank" rel="noreferrer">gpt-oss chat_template.jinja</a>' : ''} · <a href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noreferrer">InstructGPT (Ouyang et al., 2022)</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    const cb = root.querySelector('#cv-think');
    if (cb) cb.addEventListener('change', () => { thinking = cb.checked; rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).chat = { render, mount };
})();
