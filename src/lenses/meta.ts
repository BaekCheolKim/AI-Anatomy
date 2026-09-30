import type { LensId } from '../types';

export interface LensMeta {
  label: string;
  /** One line: the question this lens answers. Prose only, no numbers. */
  question: string;
}

export const LENS_META: Readonly<Record<LensId, LensMeta>> = {
  architecture: {
    label: 'Architecture',
    question: 'What does the model compute, from raw text to next-token logits, and what shape is every tensor along the way?',
  },
  inference: {
    label: 'Inference',
    question: 'How does serving work: prefill versus decode, the KV cache, sampling, and why decode waits on memory?',
  },
  training: {
    label: 'Training',
    question: 'Where do the forward pass, backward pass, gradients and optimizer states come from, and what do they cost?',
  },
  memory: {
    label: 'Memory',
    question: 'What fills GPU memory under this scenario, and at what point does it stop fitting?',
  },
  parallelism: {
    label: 'Parallelism',
    question: 'How is the same model split across GPUs by tensor, pipeline, data and expert parallelism, and what has to move between them?',
  },
  hpc: {
    label: 'HPC',
    question: 'What happens inside the GPU and between GPUs: SMs, SRAM versus HBM, links, collectives and the roofline?',
  },
};
