// Chapter: Speech recognition (encoder-decoder).
// Figures: (1) 30 s of audio through the encoder and decoder with shapes; (2) parameters by part,
// checked against the checkpoint; (3) compute and caches per 30-second window.
(function () {
  function whisperParams(w) {
    const d = w.d, f = w.ffn, lin = (i, o, b = true) => i * o + (b ? o : 0), ln = 2 * d;
    const attn = lin(d, d) + lin(d, d, false) + lin(d, d) + lin(d, d);   // q, k (no bias), v, out
    const mlp = lin(d, f) + lin(f, d);
    const enc = w.mels * d * 3 + d + d * d * 3 + d + w.srcPositions * d + w.encLayers * (attn + mlp + 2 * ln) + ln;
    const dec = w.vocab * d + w.tgtPositions * d + w.decLayers * (2 * attn + mlp + 3 * ln) + ln;   // output layer tied to the token embedding
    return { enc, dec, total: enc + dec };
  }

  function render(ctx) {
    const w = window.OTHER_MODELS.speech, P = whisperParams(w), d = w.d, out = 200;
    const encF = 2 * (P.enc - w.srcPositions * d) * w.srcPositions + w.encLayers * 4 * w.srcPositions ** 2 * d;
    const decPerTok = 2 * (P.dec - w.tgtPositions * d) + w.decLayers * 4 * w.srcPositions * d;
    const crossKV = w.decLayers * 2 * w.srcPositions * d * 2, selfKV = w.decLayers * 2 * d * 2;
    return `
      <p class="lead">Speech recognition maps audio to text. Whisper is an encoder-decoder transformer: the encoder reads a 30-second window of audio, as a spectrogram, with bidirectional attention; the decoder generates the text one token at a time, and in every layer attends both to its own earlier tokens (self-attention) and to the encoder's output (cross-attention). Longer recordings are processed window by window.</p>
      <p><b>${esc(w.name)}</b>: d = ${fmtN(d)}, ${w.encLayers} encoder and ${w.decLayers} decoder layers, ${w.heads} heads, vocabulary ${fmtN(w.vocab)}. Computed parameters: encoder ${fmtN(P.enc)}, decoder ${fmtN(P.dec)}, total ${fmtN(P.total)}${P.total === w.checkpoint ? ', equal to the checkpoint' : ` (checkpoint ${fmtN(w.checkpoint)})`}.</p>
      <h2>30 seconds of audio, step by step</h2>
      <table class="data ops"><thead><tr><th>#</th><th>step</th><th>output</th></tr></thead><tbody>
        <tr><td class="n">1</td><td><b>Resample</b> to 16 kHz</td><td class="mono shape">480,000 samples</td></tr>
        <tr><td class="n">2</td><td><b>Log-mel spectrogram</b>: 25 ms windows every 10 ms, ${w.mels} frequency bands</td><td class="mono shape">3,000 × ${w.mels}</td></tr>
        <tr><td class="n">3</td><td><b>Two convolutions</b> (width 3, the second with stride 2) and sinusoidal positions</td><td class="mono shape">${fmtN(w.srcPositions)} × ${fmtN(d)}</td></tr>
        <tr><td class="n">4</td><td><b>Encoder</b>: ${w.encLayers} layers of bidirectional attention + MLP</td><td class="mono shape">${fmtN(w.srcPositions)} × ${fmtN(d)}</td></tr>
        <tr><td class="n">5</td><td><b>Decoder</b>: special tokens for language and task, then text tokens, each layer with self-attention, cross-attention to the encoder output, MLP</td><td class="mono shape">up to ${w.tgtPositions} tokens</td></tr></tbody></table>
      <h2>Compute and caches per window</h2>
      <table class="data"><tbody>
        <tr><td>encoder, once per window</td><td class="r">${fmtFlops(encF)}</td></tr>
        <tr><td>decoder, per text token</td><td class="r">${fmtFlops(decPerTok)}</td></tr>
        <tr><td>decoder, ${out} tokens</td><td class="r">${fmtFlops(decPerTok * out)}</td></tr>
        <tr><td>cross-attention keys and values, computed once per window</td><td class="r">${fmtBytes(crossKV)} (BF16)</td></tr>
        <tr><td>self-attention cache, per text token</td><td class="r">${fmtBytes(selfKV)}</td></tr>
        <tr><td>weights, FP16</td><td class="r">${fmtBytes(P.total * 2)}</td></tr></tbody></table>
      <p class="caption">The encoder processes 1,500 positions at once and is limited by arithmetic; the decoder generates one token per step and reads its weights each time, like any decode. Cross-attention reads the same encoder output at every decoder step, so its keys and values are computed once per window and cached.</p>
      <p class="sources">Sources: <a href="${esc(w.source)}" target="_blank" rel="noreferrer">${esc(w.name)} config.json</a> · <a href="https://arxiv.org/abs/2212.04356" target="_blank" rel="noreferrer">Whisper (Radford et al., 2022)</a>.</p>`;
  }

  (window.CHAPTERS = window.CHAPTERS || {}).speech = { render };
})();
