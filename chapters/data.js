// Chapter: Data.
// Figures: (1) one web page through the filtering stages of a web-text pipeline (FineWeb's stages),
// (2) published training tokens per model, with the bytes of the token IDs.
(function () {
  const URL_EX = 'https://breadnotes.example/starter';
  // An illustrative page. Lines tagged with the stage that removes or changes them.
  const PAGE = [
    ['<html><head><title>Sourdough starter in five days</title></head><body>', 'extract'],
    ['<nav>Home · Recipes · Shop · Log in</nav>', 'extract'],
    ['<div class="cookies">We use cookies. Accept · Decline</div>', 'extract'],
    ['<article><h1>Sourdough starter in five days</h1>', 'tag'],
    ['<p>Mix 50 g of whole-wheat flour with 50 g of water in a jar and cover it loosely.</p>', 'tag'],
    ['<p>Every 24 hours, discard half of it and add 50 g of flour and 50 g of water.</p>', 'tag'],
    ['<p>By day five the starter should double in volume within six hours of feeding.</p></article>', 'tag'],
    ['<footer>Questions? Write to anna.k@breadnotes.example</footer></body></html>', 'pii'],
  ];
  const TEXT = [
    'Sourdough starter in five days',
    'Mix 50 g of whole-wheat flour with 50 g of water in a jar and cover it loosely.',
    'Every 24 hours, discard half of it and add 50 g of flour and 50 g of water.',
    'By day five the starter should double in volume within six hours of feeding.',
    'Questions? Write to anna.k@breadnotes.example',
  ];
  const STAGES = [
    ['crawl', 'Crawl', 'A web crawl stores each page as HTML. Common Crawl publishes such snapshots; FineWeb\'s first release started from 96 of them.'],
    ['url', 'URL filter', `The address ${URL_EX} is checked against block lists of domains and words. Not listed: kept.`],
    ['extract', 'Text extraction', 'An extractor keeps the main text and drops markup, navigation menus, cookie notices and other text outside the main content.'],
    ['language', 'Language', 'A classifier scores the language of the text. English score 0.97 (illustrative) against FineWeb\'s threshold of 0.65: kept.'],
    ['quality', 'Quality rules', 'Rules on word count, repeated lines, symbol-to-word ratio and lines ending in punctuation. This page passes. A page that repeats one line drops out (second document).'],
    ['dedup', 'Deduplication', 'MinHash compares word n-grams between documents. Two mirror copies of this page in the same crawl are near-duplicates: one copy is kept, the other two are dropped.'],
    ['pii', 'Personal data', 'E-mail addresses are replaced with email@example.com or firstname.lastname@example.org; public IP addresses with addresses from a fixed list of randomly generated ones. Private-range IP addresses are kept.'],
    ['tokens', 'Tokens', 'The remaining text is converted to token IDs (next chapter). Training-set sizes are counted in tokens.'],
  ];
  let stage = 0;

  function docView() {
    const id = STAGES[stage][0], idx = STAGES.findIndex((s) => s[0] === id);
    const at = (name) => idx >= STAGES.findIndex((s) => s[0] === name);
    if (!at('extract')) {
      return PAGE.map(([line]) => esc(line)).join('\n');
    }
    let lines = TEXT.map(esc);
    if (at('pii')) lines[4] = lines[4].replace('anna.k@breadnotes.example', '<span class="chg">email@example.com</span>');
    let out = lines.join('\n');
    if (id === 'quality') {
      out += '\n\n<span class="del">BEST PRICE BEST PRICE BEST PRICE\nBEST PRICE BEST PRICE BEST PRICE\nBEST PRICE BEST PRICE BEST PRICE</span>\n<span class="note-inline">second document: one line repeated, dropped</span>';
    }
    if (id === 'dedup') {
      out += `\n\n<span class="del">${lines.slice(0, 2).join('\n')} …</span>\n<span class="note-inline">mirror copy 1 of 2 at another address: near-duplicate, dropped</span>`;
    }
    if (id === 'tokens') {
      const chars = TEXT.join('\n').length;
      out += `\n\n<span class="note-inline">${chars} characters → ≈ ${Math.round(chars / 4)} tokens (about 4 characters per token for English text; exact counts depend on the tokenizer)</span>`;
    }
    return out;
  }

  function figPipeline() {
    const row = STAGES.map(([id, name], i) => `<button class="${i === stage ? 'on' : ''}" data-stage="${i}">${i + 1} ${esc(name)}</button>`).join('');
    const [, name, what] = STAGES[stage];
    return `<div class="tokens stages">${row}</div>
      <p class="status"><b>${stage + 1}. ${esc(name)}.</b> ${esc(what)}</p>
      <pre class="doc">${docView()}</pre>
      <p class="caption">One page through the stages of a web-text pipeline, in the order FineWeb applies them. The page, its address and the scores are illustrative. Struck-through text is removed at this stage; bold text is replaced. Click a stage name to move through them.</p>`;
  }

  function figTokens(m, W) {
    const known = window.MODELS.filter((x) => x.training?.tokens).sort((a, b) => a.training.tokens - b.training.tokens);
    const unknown = window.MODELS.filter((x) => !x.training?.tokens);
    const narrow = W < 560, labelW = narrow ? 0 : 168, rowH = narrow ? 40 : 28, max = Math.max(...known.map((x) => x.training.tokens)) * 1.02;
    const barW = W - labelW - (narrow ? 0 : 150);
    let s = '';
    known.forEach((mm, i) => {
      const y = i * rowH + (narrow ? 16 : 4), w = (mm.training.tokens / max) * barW, on = mm.id === m.id;
      const idBytes = mm.vocab <= 65536 ? 2 : 4;
      const val = `${fmtSig(mm.training.tokens / 1e12)}T tokens · ${fmtBytes(mm.training.tokens * idBytes)} as ${idBytes * 8}-bit IDs`;
      s += `<g class="row${on ? ' on' : ''}" data-model="${mm.id}"><title>${esc(mm.name)}: ${esc(mm.training.text)}</title>`
        + `<rect x="0" y="${y - (narrow ? 14 : 2)}" width="${W}" height="${rowH - 2}" class="hit"/>`
        + `<text x="0" y="${narrow ? y - 2 : y + 14}" class="name">${esc(mm.name)}</text>`
        + `<rect x="${labelW}" y="${y + (narrow ? 2 : 2)}" width="${w}" height="16" class="tokbar"/>`
        + `<text x="${narrow ? W : labelW + barW + 10}" y="${narrow ? y - 2 : y + 14}"${narrow ? ' text-anchor="end"' : ''} class="val">${val}</text></g>`;
    });
    const h = known.length * rowH + (narrow ? 20 : 8);
    return `${figure(s, W, h, 'published training tokens')}
      <p class="caption">Pre-training tokens as published (linear scale). A token ID is stored as a 16-bit integer when the vocabulary has at most 65,536 entries, otherwise as 32-bit. Not published by the authors: ${unknown.map((x) => esc(x.name)).join(', ')}.</p>`;
  }

  function render(ctx) {
    const m = ctx.m, W = ctx.width, tr = m.training;
    const modelLine = tr
      ? `<p><b>${esc(m.name)}</b> was pre-trained on ${esc(tr.text)} (<a href="${esc(tr.url)}" target="_blank" rel="noreferrer">source</a>). At ${m.vocab <= 65536 ? 2 : 4} bytes per token ID that is ${fmtBytes(tr.tokens * (m.vocab <= 65536 ? 2 : 4))} of token IDs, and ${fmtSig(tr.tokens / params(m).total)} tokens per stored parameter.</p>`
      : `<p>The authors of <b>${esc(m.name)}</b> have not published its number of pre-training tokens.</p>`;
    return `
      <p class="lead">Pre-training data is text, most of it from crawls of public web pages. Before training, the text is filtered: blocked addresses are dropped, the main text is extracted from the HTML, other languages and low-quality pages are removed, near-duplicate documents are dropped, and personal data such as e-mail addresses is replaced. The size of a training set is counted in tokens.</p>
      ${modelLine}
      <h2>One page through the pipeline</h2>${figPipeline()}
      <h2>Training tokens per model</h2>${figTokens(m, W)}
      <details class="more"><summary>Numbers and sources</summary><ul class="formulas">
        <li>FineWeb as first released (v1.0.0, April 2024, the version described in the paper): 15 trillion tokens of English web text, about 44 TB on disk, from 96 Common Crawl snapshots from summer 2013 to April 2024, produced with the stages above. Later releases added snapshots through June 2025, for more than 18.5 trillion tokens.</li>
        <li>Bytes of token IDs = tokens × 2 (vocabulary ≤ 65,536) or × 4. Text bytes are larger: one token covers about 4 characters of English text.</li>
        <li>Llama 3: web data with URL, document and line-level deduplication, heuristic and model-based quality filters, and a data mix tuned on small models; ${esc(window.MODELS.find((x) => x.id === 'llama-3.1-405b').training.text)} for the 405B model.</li>
      </ul></details>
      <p class="sources">Sources: <a href="https://arxiv.org/abs/2406.17557" target="_blank" rel="noreferrer">The FineWeb Datasets (Penedo et al., 2024)</a> · <a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noreferrer">Llama 3 paper, section 3.1</a> · <a href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noreferrer">DeepSeek-V3 report</a>${tr ? ` · <a href="${esc(tr.url)}" target="_blank" rel="noreferrer">${esc(m.name)} tokens</a>` : ''}.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelectorAll('[data-stage]').forEach((b) => b.addEventListener('click', () => { stage = Number(b.dataset.stage); rerender(); }));
  }

  (window.CHAPTERS = window.CHAPTERS || {}).data = { render, mount };
})();
