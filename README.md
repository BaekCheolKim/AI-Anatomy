# AI-Anatomy
Visualizing how AI models work under the hood.

Open `index.html` in a browser (double-click it). No install or build needed.

The page shows how a transformer LLM turns a text prompt into the next token, on one screen:
tokenizer → embedding → transformer layer (attention, MLP with its activation, residual stream) → LM head → softmax → sampling.
Click any box for an explanation, its tensor shapes, and how it connects to other parts.
