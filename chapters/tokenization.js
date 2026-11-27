// Chapter: Tokenization.
// Figures: (1) byte-pair encoding trained live on editable text, one merge per step, with the merge
// list and tokens-vs-merges curve; (2) vocabulary sizes and embedding/output-layer parameters per model.
(function () {
  const SAMPLE = 'Tokenizers split text into tokens. Frequent strings such as "the", "token" and "ing" become single tokens; rare words are split into pieces, down to single bytes. The tokenizer is trained once, before the model, and then stays fixed.';
  let text = SAMPLE, step = 24, trained = null;

  // GPT-2's pre-tokenization pattern: merges never cross these pieces, and a space attaches to the
  // start of the following word.
  const PRE = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const enc = new TextEncoder(), dec = new TextDecoder('utf-8', { fatal: false });

  // Byte-level BPE: start from UTF-8 bytes (IDs 0–255); repeatedly merge the most frequent adjacent
  // pair (first occurrence wins ties) into a new ID, while some pair occurs at least twice.
  function train(src, maxMerges = 400) {
    let seqs = (src.match(PRE) || []).map((w) => Array.from(enc.encode(w)));
    const bytesOf = new Map(range(256).map((b) => [b, [b]]));
    const merges = [], snapshots = [seqs];
    for (let i = 0; i < maxMerges; i++) {
      const counts = new Map();
      seqs.forEach((s) => { for (let j = 0; j + 1 < s.length; j++) { const k = `${s[j]},${s[j + 1]}`; counts.set(k, (counts.get(k) || 0) + 1); } });
      let best = null, bestN = 1;
      counts.forEach((n, k) => { if (n > bestN) { best = k; bestN = n; } });
      if (!best) break;
      const [a, b] = best.split(',').map(Number), id = 256 + i;
      bytesOf.set(id, [...bytesOf.get(a), ...bytesOf.get(b)]);
      merges.push({ a, b, id, n: bestN });
      seqs = seqs.map((s) => { const o = []; for (let j = 0; j < s.length; j++) { if (j + 1 < s.length && s[j] === a && s[j + 1] === b) { o.push(id); j++; } else o.push(s[j]); } return o; });
      snapshots.push(seqs);
    }
    return { merges, snapshots, bytesOf, nBytes: enc.encode(src).length };
  }

  const show = (bytes) => {
    const s = dec.decode(new Uint8Array(bytes));
    return s.includes('�') ? bytes.map((b) => `⟨${b.toString(16).padStart(2, '0')}⟩`).join('') : esc(s).replace(/ /g, '␣').replace(/\n/g, '↵');
  };

  function curve(W) {
    const counts = trained.snapshots.map((sn) => sn.reduce((s, x) => s + x.length, 0));
    const n = counts.length - 1, H = 110, padL = 40, padB = 22;
    const sx = (i) => padL + (n ? (i / n) * (W - padL - 8) : 0), sy = (v) => 6 + (1 - v / counts[0]) * (H - padB - 6);
    let s = `<line x1="${padL}" x2="${W - 8}" y1="${H - padB}" y2="${H - padB}" class="grid"/>`;
    s += `<polyline points="${counts.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ')}" class="line-ink"/>`;
    s += `<circle cx="${sx(step)}" cy="${sy(counts[step])}" r="4" class="dot-ink"/>`;
    s += `<text x="${padL - 6}" y="${sy(counts[0]) + 4}" text-anchor="end" class="axis">${counts[0]}</text><text x="${padL - 6}" y="${sy(counts[n]) + 4}" text-anchor="end" class="axis">${counts[n]}</text>`;
    s += `<text x="${padL}" y="${H - 6}" class="axis">0 merges</text><text x="${W - 8}" y="${H - 6}" text-anchor="end" class="axis">${n} merges</text>`;
    return figure(s, W, H, 'tokens versus merges');
  }

  function output(W) {
    const t = trained, seqs = t.snapshots[step], tokens = seqs.flat();
    const shown = seqs.map((s) => s.map((id) => `<span class="tk" title="ID ${id}">${show(t.bytesOf.get(id))}</span>`).join('')).join('');
    const list = t.merges.slice(0, step).slice(-12).map((mg, i, arr) => {
      const k = step - arr.length + i + 1;
      return `<li>${k}. <code>${show(t.bytesOf.get(mg.a))}</code> + <code>${show(t.bytesOf.get(mg.b))}</code> → <code>${show(t.bytesOf.get(mg.id))}</code> (ID ${mg.id}, ${mg.n} times)</li>`;
    }).join('');
    return `<p class="status">${step} merge${step === 1 ? '' : 's'}: vocabulary 256 + ${step} = ${256 + step} entries. This text: ${t.nBytes} bytes → <b>${tokens.length} tokens</b> (${(t.nBytes / tokens.length).toFixed(2)} bytes per token).</p>
      <div class="tkline">${shown}</div>
      <p class="tkids"><span class="note-inline">token IDs:</span> ${tokens.slice(0, 60).join(' ')}${tokens.length > 60 ? ' …' : ''}</p>
      ${curve(W)}
      ${step ? `<p class="caption">Last merges${step > 12 ? ` (${step - 12} earlier ones not listed)` : ''}:</p><ol class="merges">${list}</ol>` : ''}`;
  }

  function figVocab(m, W) {
    const models = window.MODELS.slice().sort((a, b) => a.vocab - b.vocab);
    const narrow = W < 560, labelW = narrow ? 0 : 168, rowH = narrow ? 40 : 28, barW = W - labelW - (narrow ? 0 : 170), max = 250000;
    let s = '';
    models.forEach((mm, i) => {
      const P = params(mm), y = i * rowH + (narrow ? 16 : 4), on = mm.id === m.id;
      const embOut = P.embed + P.head;
      const val = `${fmtN(mm.vocab)} · ${fmtParams(embOut)} parameters (${fmtPct(embOut, P.total)})`;
      s += `<g class="row${on ? ' on' : ''}" data-model="${mm.id}"><title>${esc(mm.name)}: ${esc(mm.tokenizer)}, ${fmtN(mm.vocab)} entries; embedding${mm.tied ? ' (also the output layer)' : ' and output layer'} ${fmtN(embOut)} parameters</title>`
        + `<rect x="0" y="${y - (narrow ? 14 : 2)}" width="${W}" height="${rowH - 2}" class="hit"/>`
        + `<text x="0" y="${narrow ? y - 2 : y + 14}" class="name">${esc(mm.name)}</text>`
        + `<rect x="${labelW}" y="${y + 2}" width="${(mm.vocab / max) * barW}" height="16" class="tokbar"/>`
        + `<text x="${narrow ? W : labelW + barW + 10}" y="${narrow ? y - 2 : y + 14}"${narrow ? ' text-anchor="end"' : ''} class="val">${val}</text></g>`;
    });
    return `${figure(s, W, models.length * rowH + (narrow ? 20 : 8), 'vocabulary sizes')}
      <p class="caption">Vocabulary entries (bar) and the parameters of the embedding table plus the output layer, each V × H (share of all parameters). GPT-2 uses one matrix for both. Click a row to select that model.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, P = params(m), W = ctx.width;
    if (!trained) trained = train(text);
    step = Math.min(step, trained.merges.length);
    const emb = m.tied
      ? `Its embedding table is ${fmtN(m.vocab)} × ${fmtN(m.hidden)} = ${fmtParams(P.embed)} parameters; the same matrix computes the output scores (tied).`
      : `Its embedding table is ${fmtN(m.vocab)} × ${fmtN(m.hidden)} = ${fmtParams(P.embed)} parameters, and the output layer has the same shape: together ${fmtPct(P.embed + P.head, P.total)} of the model.`;
    return `
      <p class="lead">The network reads integers, not characters. A tokenizer maps text to a sequence of token IDs, each an index into a fixed vocabulary. Byte-level byte-pair encoding (BPE) builds that vocabulary: it starts from the 256 possible byte values and repeatedly adds the most frequent adjacent pair as a new entry, so frequent strings become one token and rare strings stay several. Llama 2's SentencePiece BPE starts from Unicode characters instead and uses byte tokens only for characters outside its vocabulary.</p>
      <p><b>${esc(m.name)}</b> uses ${esc(m.tokenizer)} with ${fmtN(m.vocab)} entries. ${emb}</p>
      <h2>Byte-pair encoding, one merge at a time</h2>
      <p class="controls"><label for="bpe-text">Text (edit it):</label></p>
      <textarea id="bpe-text" rows="4" spellcheck="false">${esc(text)}</textarea>
      <p class="controls"><label>Merges <input id="bpe-step" type="range" min="0" max="${trained.merges.length}" value="${step}"></label> <span id="bpe-max">of ${trained.merges.length} possible on this text</span></p>
      <div id="bpe-out">${output(W)}</div>
      <p class="caption">Merges are counted inside pieces split by GPT-2's pre-tokenization pattern, so no token spans two words and a space attaches to the start of the next word. Training stops when no pair occurs twice. Production tokenizers run the same procedure on far more text until the vocabulary reaches its target size: GPT-2's 50,257 entries are 256 bytes, 50,000 merges and one end-of-text token. Hover a token for its ID; ⟨xx⟩ marks a byte that is part of a multi-byte character.</p>
      <p>The network receives one ID per token, not the letters inside it. Questions about letters, such as spelling a word or counting a character in it, depend on what the model has learned about each token's letters from the training data.</p>
      <h2>Vocabularies of the models</h2>${figVocab(m, W)}
      <p class="sources">Sources: <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank" rel="noreferrer">GPT-2 paper, section 2.2 (byte-level BPE)</a> · <a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noreferrer">Llama 3 paper (128K vocabulary)</a> · ${configLink(m)}.</p>`;
  }

  function mount(root, ctx) {
    const out = root.querySelector('#bpe-out'), slider = root.querySelector('#bpe-step'), max = root.querySelector('#bpe-max');
    const refresh = () => { out.innerHTML = output(ctx.width); };
    slider.addEventListener('input', () => { step = Number(slider.value); refresh(); });
    let timer = 0;
    root.querySelector('#bpe-text').addEventListener('input', (e) => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        text = e.target.value || ' ';
        trained = train(text);
        slider.max = trained.merges.length;
        step = Math.min(step, trained.merges.length);
        slider.value = step;
        max.textContent = `of ${trained.merges.length} possible on this text`;
        refresh();
      }, 150);
    });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).tokenization = { render, mount };
})();
