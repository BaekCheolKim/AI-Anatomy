// Chapter: From request to response.
// Figure: the path of one chat request through an inference server, step by step, with where each
// step runs and what it produces for the selected model.
(function () {
  function render(ctx) {
    const m = ctx.m, kv = kvBytesPerToken(m), P = params(m);
    const steps = [
      ['HTTP request', 'CPU', 'An OpenAI-compatible API call: a list of messages, sampling settings (temperature, top-p, maximum tokens), optional tools and an output schema.'],
      ['Chat template', 'CPU', `The messages are written as one string with the model's special tokens (${m.chat ? esc(m.chat) : 'none: a base model takes plain text'}); see Conversations as tokens.`],
      ['Tokenize', 'CPU', `The string becomes token IDs, ${fmtN(m.vocab)} possible values.`],
      ['Schedule', 'CPU', 'The request waits until the KV pool has blocks for it; prefix blocks already cached are reused.'],
      ['Prefill', 'GPU', `All prompt tokens in one pass (or in chunks); their keys and values are written to the cache: ${fmtBytes(kv)} per token for ${esc(m.name)}.`],
      ['Sample', 'GPU', 'The last position\'s logits pass through temperature and top-p; with an output schema, tokens that would break it are masked to −∞ first (structured output).'],
      ['Decode loop', 'GPU', `One token per step: ${fmtFlops(2 * P.matmul)} of weight multiplications plus attention over the cache, until an end token or the maximum length.`],
      ['Detokenize and stream', 'CPU', 'Each new token is turned back into text and sent to the client as soon as it is complete (server-sent events), so text appears while the rest is generated.'],
      ['Free', 'CPU', 'The scheduler returns the request\'s blocks to the free queue of the pool in GPU memory; blocks with a prefix hash keep their contents and can be reused by later requests until they are evicted.'],
    ];
    return `
      <p class="lead">A chat request passes through software on the CPU and the model on the GPUs. The CPU side turns messages into token IDs, schedules the request and turns generated IDs back into text; the GPU side runs prefill once, which yields the first token, then one decode step per further token. Time to first token covers everything up to the first sampled token; after that, tokens arrive once per decode step.</p>
      <h2>The path of one request</h2>
      <table class="data ops"><thead><tr><th>#</th><th>step</th><th>runs on</th><th>what happens</th></tr></thead><tbody>
        ${steps.map(([name, where, what], i) => `<tr><td class="n">${i + 1}</td><td><b>${name}</b></td><td>${where}</td><td class="f">${what}</td></tr>`).join('')}</tbody></table>
      <details class="more"><summary>Latency</summary><ul class="formulas">
        <li>Time to first token = queueing + tokenization + prefill (grows with prompt length) + one sampling step.</li>
        <li>Time per output token = one decode step of the batch the request runs in.</li>
        <li>Total ≈ time to first token + (output tokens − 1) × time per output token.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://platform.openai.com/docs/api-reference/chat" target="_blank" rel="noreferrer">OpenAI Chat Completions API</a> · <a href="https://arxiv.org/abs/2307.09702" target="_blank" rel="noreferrer">grammar-constrained decoding (Outlines)</a>.</p>`;
  }

  (window.CHAPTERS = window.CHAPTERS || {}).request = { render };
})();
