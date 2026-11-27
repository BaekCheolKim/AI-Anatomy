// Chapter: Embedding models and retrieval.
// Figures: (1) the retrieval pipeline with sizes; (2) index size for a document collection at chosen
// dimensions and precision.
(function () {
  let docs = 1e6, dims = 4096, vbytes = 4;

  function render(ctx) {
    const E = window.OTHER_MODELS.embedding, m = { ...E.llm, id: E.id, name: E.name }, P = params(m), lm = ctx.m;
    const ok = P.total === E.checkpoint;
    const index = docs * dims * vbytes, chunk = 512;
    const embedFlops = 2 * P.matmul * chunk * docs;
    return `
      <p class="lead">An embedding model maps a text to one vector, so that texts with related meaning get vectors with a large cosine similarity. It is trained with a contrastive loss, which increases the cosine similarity of a query with its matching passage and decreases it with non-matching passages. Retrieval uses it to find documents for a question: documents are embedded once and stored; at question time the question is embedded and the nearest document vectors are looked up; the matching texts are placed in the language model's context (retrieval-augmented generation).</p>
      <p><b>${esc(E.name)}</b>: a ${E.llm.layers}-layer Qwen3 decoder (H = ${fmtN(E.llm.hidden)}) without the output layer. The final hidden state of the last token, L2-normalized, is the embedding: up to ${fmtN(E.dims)} values, and any first ${E.minDims} to ${fmtN(E.dims)} of them can be used (trained with Matryoshka representation learning), up to ${fmtN(E.maxLen)} tokens of input. Queries are prefixed with a one-line task instruction. Parameters: ${fmtN(P.total)} computed${ok ? ', equal to the checkpoint' : ` (checkpoint: ${fmtN(E.checkpoint)})`}.</p>
      <h2>Retrieval, step by step</h2>
      <table class="data ops"><thead><tr><th>#</th><th>step</th><th>when</th><th>cost for this setup</th></tr></thead><tbody>
        <tr><td class="n">1</td><td><b>Split documents into chunks</b> of about ${chunk} tokens</td><td>once</td><td>${fmtN(docs)} chunks</td></tr>
        <tr><td class="n">2</td><td><b>Embed every chunk</b>: one forward pass, keep the last token's hidden state</td><td>once</td><td>${fmtFlops(embedFlops)} (2 × ${fmtParams(P.matmul)} × ${chunk} tokens × ${fmtParams(docs)} chunks)</td></tr>
        <tr><td class="n">3</td><td><b>Store the vectors</b> in an index (exact or approximate nearest-neighbour search)</td><td>once</td><td>${fmtBytes(index)}</td></tr>
        <tr><td class="n">4</td><td><b>Embed the question</b></td><td>per question</td><td>${fmtFlops(2 * P.matmul * 64)} for 64 tokens</td></tr>
        <tr><td class="n">5</td><td><b>Find the nearest vectors</b> by cosine similarity</td><td>per question</td><td>${fmtFlops(2 * dims * docs)} for an exact search over all vectors</td></tr>
        <tr><td class="n">6</td><td><b>Put the top chunks into the prompt</b> of the language model</td><td>per question</td><td>for example 5 × ${chunk} tokens: ${fmtBytes(5 * chunk * kvBytesPerToken(lm))} of KV cache for ${esc(lm.name)}</td></tr></tbody></table>
      <p class="controls"><label class="nw">chunks <select id="em-n">${[1e4, 1e6, 1e8].map((v) => `<option value="${v}"${v === docs ? ' selected' : ''}>${fmtParams(v)}</option>`).join('')}</select></label> <label class="nw">dimensions <select id="em-d">${[256, 1024, 4096].map((v) => `<option${v === dims ? ' selected' : ''}>${v}</option>`).join('')}</select></label> <label class="nw">stored as <select id="em-b"><option value="4"${vbytes === 4 ? ' selected' : ''}>FP32</option><option value="2"${vbytes === 2 ? ' selected' : ''}>FP16</option><option value="1"${vbytes === 1 ? ' selected' : ''}>INT8</option></select></label></p>
      <p class="caption">Index size = chunks × dimensions × bytes per value. Fewer dimensions and lower precision shrink the index and the search, at some loss of retrieval accuracy.</p>
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Cosine similarity: cos(q, d) = q·d / (‖q‖‖d‖); with normalized vectors, the dot product.</li>
        <li>Contrastive (InfoNCE) loss for a query q with matching passage d⁺ and non-matching passages d<sub>j</sub>: −ln [exp(cos(q, d⁺)/τ) / (exp(cos(q, d⁺)/τ) + Σ<sub>j</sub> exp(cos(q, d<sub>j</sub>)/τ))], with temperature τ.</li>
      </ul></details>
      <p class="sources">Sources: <a href="${esc(E.source)}" target="_blank" rel="noreferrer">${esc(E.name)} config.json</a> · <a href="${esc(E.card)}" target="_blank" rel="noreferrer">model card</a> · <a href="https://arxiv.org/abs/2005.11401" target="_blank" rel="noreferrer">retrieval-augmented generation (Lewis et al.)</a> · <a href="https://arxiv.org/abs/2205.13147" target="_blank" rel="noreferrer">Matryoshka representation learning</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#em-n').addEventListener('change', (e) => { docs = Number(e.target.value); rerender(); });
    root.querySelector('#em-d').addEventListener('change', (e) => { dims = Number(e.target.value); rerender(); });
    root.querySelector('#em-b').addEventListener('change', (e) => { vbytes = Number(e.target.value); rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).embeddings = { render, mount };
})();
