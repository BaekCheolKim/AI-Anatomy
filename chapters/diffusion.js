// Chapter: Image generation (diffusion / flow matching).
// Figures: (1) one image as latent tokens; (2) parameters of the denoising transformer, checked
// against the checkpoint; (3) FLOPs and memory per image, compared with text generation.
(function () {
  let size = 1024, steps = 4;

  // FLUX transformer: double-stream blocks (separate weights for image and text tokens, joint
  // attention) and single-stream blocks (one set of weights over the joined sequence).
  function fluxParams(t) {
    const d = t.heads * t.headDim, lin = (i, o) => i * o + o, mlp = t.mlpRatio * d;
    const double = 2 * lin(d, 6 * d) + 6 * lin(d, d) + 2 * lin(d, d) + 4 * t.headDim + 2 * (lin(d, mlp) + lin(mlp, d));
    const single = lin(d, 3 * d) + lin(d, mlp) + 3 * lin(d, d) + 2 * t.headDim + lin(d + mlp, d);
    const embed = lin(t.inChannels, d) + lin(t.jointDim, d) + lin(256, d) + lin(d, d) + lin(t.pooledDim, d) + lin(d, d);
    const out = lin(d, 2 * d) + lin(d, t.inChannels);
    return { double, single, embed, out, total: t.double * double + t.single * single + embed + out };
  }

  function render(ctx) {
    const D = window.OTHER_MODELS.diffusion, t = D.transformer, P = fluxParams(t), lm = ctx.m;
    const lat = size / D.vaeDown, imgTok = (lat / D.patch) ** 2, seq = imgTok + D.textTokens, d = t.heads * t.headDim;
    const perStep = 2 * (P.total - P.embed) * seq + (t.double + t.single) * 4 * seq * seq * d, total = perStep * steps;
    const tokEq = total / (2 * params(lm).matmul);
    return `
      <p class="lead">Image generators of this kind learn to remove noise. Training takes an image, encodes it into a smaller latent with an autoencoder (VAE), mixes the latent with random noise at a random level, and trains a transformer to predict the direction from the noisy latent back to the clean one, given the text. Generation starts from pure noise and applies the transformer for a number of steps, each moving the latent a little toward an image that matches the text; the VAE decoder then turns the latent into pixels. Every step runs the whole transformer over all image tokens; there is no KV cache.</p>
      <p><b>${esc(D.name)}</b>: a ${fmtParams(P.total)}-parameter transformer with ${t.double} double-stream blocks (separate weights for image and text tokens, attention over both) and ${t.single} single-stream blocks (one set of weights over the joined sequence), width ${fmtN(d)} (${t.heads} heads of ${t.headDim}). Computed parameters: ${fmtN(P.total)}${P.total === D.checkpoint ? ', equal to the checkpoint' : ` (checkpoint ${fmtN(D.checkpoint)})`}. Text enters through a T5 text encoder (${D.textTokens} tokens) and a CLIP text encoder (one pooled vector). Trained for 1 to 4 steps.</p>
      <p class="controls"><label class="nw">image <select id="df-s">${[512, 1024, 2048].map((v) => `<option value="${v}"${v === size ? ' selected' : ''}>${v} × ${v}</option>`).join('')}</select></label> <label class="nw">steps <select id="df-n">${[1, 4, 28, 50].map((v) => `<option${v === steps ? ' selected' : ''}>${v}</option>`).join('')}</select></label></p>
      <h2>One image as tokens</h2>
      <table class="data"><tbody>
        <tr><td>pixels</td><td class="mono">${size} × ${size} × 3</td></tr>
        <tr><td>VAE latent (÷ ${D.vaeDown} per side)</td><td class="mono">${lat} × ${lat} × ${D.latentChannels}</td></tr>
        <tr><td>tokens: ${D.patch} × ${D.patch} latent patches</td><td class="mono">${fmtN(imgTok)} × ${D.patch * D.patch * D.latentChannels}</td></tr>
        <tr><td>with the text tokens</td><td class="mono">${fmtN(seq)} positions × ${fmtN(d)}</td></tr></tbody></table>
      <h2>Compute and memory per image</h2>
      <table class="data"><tbody>
        <tr><td>FLOPs per step</td><td class="r">${fmtFlops(perStep)} (weights ${fmtFlops(2 * (P.total - P.embed) * seq)}, attention over ${fmtN(seq)} positions ${fmtFlops((t.double + t.single) * 4 * seq * seq * d)})</td></tr>
        <tr><td>FLOPs per image, ${steps} step${steps > 1 ? 's' : ''}</td><td class="r"><b>${fmtFlops(total)}</b></td></tr>
        <tr><td>the same FLOPs with ${esc(lm.name)}</td><td class="r">about ${fmtN(tokEq)} generated tokens</td></tr>
        <tr><td>transformer weights, BF16</td><td class="r">${fmtBytes(P.total * 2)}, read once per step</td></tr></tbody></table>
      <p class="caption">Attention grows with the square of the number of image tokens, which grows with the square of the image side: doubling the side multiplies the attention FLOPs by 16. Each step processes all tokens at once, so image generation is limited by arithmetic, like prefill, not by memory bandwidth like decode. The text encoders and the VAE add a few gigabytes of weights and run once per image.</p>
      <details class="more"><summary>Formulas</summary><ul class="formulas">
        <li>Rectified flow: x<sub>t</sub> = (1 − t)·x<sub>0</sub> + t·ε with t ∈ [0, 1] and noise ε; the network predicts v ≈ ε − x<sub>0</sub>; loss ‖v<sub>θ</sub>(x<sub>t</sub>, t, text) − (ε − x<sub>0</sub>)‖².</li>
        <li>Generation: x ← x − Δt·v<sub>θ</sub>(x, t, text) from t = 1 to t = 0 in N steps. A model distilled for few steps (schnell) takes 1–4 steps; others take 20–50.</li>
      </ul></details>
      <p class="sources">Sources: <a href="${esc(D.source)}" target="_blank" rel="noreferrer">${esc(D.name)}</a> (${esc(D.sourceNote)}) · <a href="https://arxiv.org/abs/2403.03206" target="_blank" rel="noreferrer">rectified-flow transformers (SD3)</a> · <a href="https://arxiv.org/abs/2112.10752" target="_blank" rel="noreferrer">latent diffusion</a>.</p>`;
  }

  function mount(root, ctx, rerender) {
    root.querySelector('#df-s').addEventListener('change', (e) => { size = Number(e.target.value); rerender(); });
    root.querySelector('#df-n').addEventListener('change', (e) => { steps = Number(e.target.value); rerender(); });
  }

  (window.CHAPTERS = window.CHAPTERS || {}).diffusion = { render, mount };
})();
