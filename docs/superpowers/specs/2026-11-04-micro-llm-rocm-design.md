# Micro LLM for Linux + AMD RX 6800 XT (ROCm): Design

## Intent

Build a small language model from scratch for research and study. Train it on a Linux machine with an AMD RX 6800 XT (16 GB VRAM, RDNA2, gfx1030) using ROCm. The output is a single checkpoint file (plus a safetensors export) that can also be run for generation on the user's Mac. The first prototype (a 2.7M-parameter character-level GPT trained on the Mac) was too heavy for that machine, which is why training moves to the GPU box. The prototype files have been deleted.

## Goals

- Train a ~30M-parameter decoder-only transformer, Llama-style (RoPE, RMSNorm, SwiGLU, weight-tied embeddings), readable enough to study line by line.
- Use a trained byte-level BPE tokenizer (8192 vocab) instead of characters, so the model produces words rather than letter soup.
- Train on TinyStories, which small models learn to write coherent text from.
- Run well on the 6800 XT: fp16 autocast with a gradient scaler, fused attention via PyTorch `scaled_dot_product_attention`, gradient accumulation, resumable checkpoints.
- The same code runs on CPU and Apple MPS for generation and tests, with device auto-detection.
- A README runbook so the user can set up ROCm PyTorch on the other machine and run training with copy-paste commands.

## Non-goals

- No multi-GPU, no distributed training.
- No fine-tuning, instruction-tuning, or RLHF.
- No GGUF/llama.cpp export (possible later).
- No `torch.compile` by default (flaky on ROCm for this card); it is an opt-in flag only.
- No web UI or serving.

## Anti-goals

- Training that only works on the author's machine: the setup must be reproducible from the README alone.
- Silent CPU fallback on the GPU box: if the user asks for a GPU run and none is visible, fail loudly.
- A model that "trains" but whose checkpoint cannot be loaded on the Mac.

## Constraints

- Target: Linux, RX 6800 XT, ROCm. gfx1030 is officially supported, so no `HSA_OVERRIDE_GFX_VERSION` should be needed. Mention it only as a troubleshooting step.
- PyTorch ROCm wheels. The ROCm version is chosen by the builder from what the PyTorch site currently offers. The setup script takes it as a parameter. (Builder's call.)
- On ROCm the PyTorch device string is still `"cuda"`. Detect ROCm via `torch.version.hip`.
- RDNA2 has no fast bf16: use fp16 autocast + `torch.amp.GradScaler`, not bf16.
- Python 3.10+. Dependencies kept to: `torch`, `numpy`, `tokenizers`, `datasets`, `safetensors`, `pytest`.
- Model defaults: dim 512, 8 layers, 8 heads, SwiGLU hidden 1408, context 512, vocab 8192, tied embeddings (~30M parameters).
- Checkpoint is portable: loads with `map_location` on any device, stores config and tokenizer path/hash.

## Decisions made by the author (not by the user)

All of the above is the author's choice. The user said only: redesign and plan for Linux + 6800 XT + ROCm, and run it on another machine. Model size, dataset, tokenizer, and Llama-style architecture are proposals the user can change.

## Left to the builder

Exact PyTorch ROCm wheel version, learning-rate schedule details, batch size and accumulation steps (target: ~64k tokens per optimizer step, fit in 16 GB), logging format, and sampling defaults.

## Success looks like

1. `scripts/check_gpu.py` on the GPU box prints the AMD GPU name and ROCm/HIP version and runs a matmul on it.
2. A short smoke training run (200 steps) shows decreasing loss and reports tokens/sec and peak VRAM.
3. A full run produces `checkpoints/micro-llm.pt` and `checkpoints/micro-llm.safetensors`; generated text from a prompt like "Once upon a time" is coherent short-story English.
4. The checkpoint copied to the Mac generates text there with `python -m microllm.generate`.
