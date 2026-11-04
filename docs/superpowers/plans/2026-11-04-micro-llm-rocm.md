# Micro LLM (ROCm / RX 6800 XT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ~30M-parameter Llama-style GPT, trained on TinyStories on a Linux RX 6800 XT with ROCm, producing a checkpoint that also generates text on a Mac.

**Architecture:** A small `microllm` package: config, model, tokenizer, data prep, trainer, generator, export. Training uses fp16 autocast + GradScaler and SDPA attention. Everything is device-agnostic except an explicit `--require-gpu` flag for the GPU box.

**Tech Stack:** Python 3.10+, PyTorch (ROCm wheels on the GPU box), `tokenizers`, `datasets`, `safetensors`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-04-micro-llm-rocm-design.md`

## Global Constraints

- Target hardware: Linux, AMD RX 6800 XT (gfx1030), ROCm. Device string stays `"cuda"`; ROCm detected via `torch.version.hip`.
- Mixed precision is fp16 autocast + `torch.amp.GradScaler`. No bf16.
- Defaults: dim 512, n_layer 8, n_head 8, SwiGLU hidden 1408, context 512, vocab 8192, tied embeddings.
- Architecture: RoPE, RMSNorm, SwiGLU, no biases, causal attention via `F.scaled_dot_product_attention(..., is_causal=True)`.
- Dependencies limited to `torch`, `numpy`, `tokenizers`, `datasets`, `safetensors`, `pytest`.
- No `torch.compile` unless `--compile` is passed.
- Checkpoints load with `map_location` on any device (CPU, MPS, CUDA/ROCm).
- The Mac prototype files were deleted; the repo starts clean apart from `docs/` and a local `.venv` (CPU/MPS torch, usable for running tests on the Mac).
- All tests must pass on CPU (the Mac); GPU-only checks are marked `@pytest.mark.gpu` and skipped when no GPU is visible.

## Review Focus

- Prompt text containing characters or emoji never seen in training: the byte-level tokenizer must encode and decode it without error (Task 3).
- Prompt longer than the context window (512 tokens): generation must crop, not crash (Task 5).
- `--require-gpu` on a machine with no GPU: exit non-zero with a clear message, never silently use CPU (Task 1 and Task 4).
- Resuming training from a checkpoint: step counter, optimizer, scaler, and LR position must continue, with no loss spike (Task 4).
- fp16 overflow producing NaN/inf loss: trainer must skip the step via GradScaler and abort with a message after repeated non-finite losses (Task 4).

---

### Task 1: Scaffold and GPU check

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `scripts/setup_rocm.sh`, `scripts/check_gpu.py`, `microllm/__init__.py`, `microllm/device.py`, `tests/test_device.py`
- Run once: `git init`

**Interfaces:**
- Produces: `microllm.device.pick_device(require_gpu: bool = False) -> str` returns `"cuda"` (NVIDIA or ROCm), `"mps"`, or `"cpu"`; raises `RuntimeError("--require-gpu set but no GPU visible ...")` when `require_gpu` and result would be `"mps"` or `"cpu"`.
- Produces: `microllm.device.describe(device: str) -> str` returns human text, including `HIP <version>` when `torch.version.hip` is set.

- [ ] **Step 1: Write failing tests** in `tests/test_device.py`: `test_pick_device_returns_known_value` (result in `{"cuda","mps","cpu"}`), `test_require_gpu_raises_when_no_gpu` (monkeypatch `torch.cuda.is_available` and `torch.backends.mps.is_available` to `False`; `pytest.raises(RuntimeError, match="require-gpu")`), `test_describe_mentions_hip` (monkeypatch `torch.version.hip="6.2"` with device `"cuda"` and a stubbed device name; output contains `"HIP 6.2"`).
- [ ] **Step 2: Run** `pytest tests/test_device.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `device.py`** per Interfaces. `pyproject.toml` declares the package and dependencies (torch is installed by the setup script, not pinned in `dependencies`).
- [ ] **Step 4: Run** `pytest tests/test_device.py -v`. Expected: PASS.
- [ ] **Step 5: Write `scripts/setup_rocm.sh <rocm-version>`**: creates `.venv`, installs PyTorch from `https://download.pytorch.org/whl/rocm<version>`, then `pip install -e .`. It first prints a reminder that the user must be in the `render` and `video` groups. Write `scripts/check_gpu.py`: calls `pick_device(require_gpu=True)`, prints `describe()`, runs a 4096x4096 fp16 matmul on the device and prints elapsed ms. Exit 1 with message if no GPU.
- [ ] **Step 6: Commit** `git add -A && git commit -m "chore: scaffold microllm package, GPU check"`

### Task 2: Model

**Files:**
- Create: `microllm/config.py`, `microllm/model.py`, `tests/test_model.py`

**Interfaces:**
- Produces: `@dataclass Config(vocab_size=8192, block_size=512, n_layer=8, n_head=8, dim=512, ffn_hidden=1408, dropout=0.0)` with `to_dict()` and `Config.from_dict(d)`.
- Produces: `class MicroLLM(nn.Module)` with `forward(idx: LongTensor[B,T], targets: LongTensor[B,T] | None = None) -> tuple[Tensor, Tensor | None]` (logits `[B,T,vocab]`, loss), and `num_params() -> int`.
- Produces: `generate(model, idx, max_new_tokens, temperature=0.8, top_k=40) -> LongTensor` in `microllm/model.py`, cropping context to `block_size`.

- [ ] **Step 1: Write failing tests**: `test_logits_shape` (tiny config dim 64, 2 layers, vocab 100; output `[2,16,100]`), `test_causality` (changing token at position 10 does not change logits at positions < 10), `test_loss_is_finite_scalar`, `test_weight_tying` (`model.head.weight is model.tok.weight`), `test_default_param_count_in_range` (default config total in 25M to 35M), `test_config_roundtrip`.
- [ ] **Step 2: Run** `pytest tests/test_model.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `Config` and `MicroLLM` (modules: `RMSNorm`, rotary embedding applied to q/k, `Attention` using SDPA with `is_causal=True`, `SwiGLU`, pre-norm `Block`). Init: normal std 0.02, residual output projections scaled by `1/sqrt(2*n_layer)`.
- [ ] **Step 4: Run** `pytest tests/test_model.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: Llama-style MicroLLM model"`

### Task 3: Tokenizer and data prep

**Files:**
- Create: `microllm/tokenizer.py`, `microllm/data.py`, `scripts/prepare_data.py`, `tests/test_tokenizer.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str) -> None` (byte-level BPE via `tokenizers`, special token `<|endoftext|>`); `class Tok` with `Tok.load(path)`, `encode(text: str) -> list[int]`, `decode(ids: list[int]) -> str`, `vocab_size: int`, `eot_id: int`.
- Produces: `tokenize_to_bin(texts: Iterable[str], tok: Tok, out_path: str) -> int` writes uint16 tokens (each document followed by `eot_id`), returns token count; `class BinDataset(path: str, block_size: int)` with `get_batch(batch_size: int, device: str, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]` returning `x, y` of shape `[batch_size, block_size]` with `y` shifted by one.
- `scripts/prepare_data.py --max-docs N` downloads `roneneldan/TinyStories` via `datasets`, trains the tokenizer on the train split, writes `data/train.bin`, `data/val.bin`, `data/tokenizer.json`.

- [ ] **Step 1: Write failing tests**: `test_roundtrip_ascii`, `test_roundtrip_unseen_unicode` (encode and decode `"héllo 🙂 日本"` returns identical string), `test_vocab_size_is_8192_when_requested` (train on a small repeated corpus with vocab 300 and assert `<= 300`), `test_bin_has_eot_between_docs`, `test_get_batch_shift` (`y[:, :-1] == x[:, 1:]`), `test_get_batch_shapes_and_dtype` (long dtype).
- [ ] **Step 2: Run** `pytest tests/test_tokenizer.py tests/test_data.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces. `BinDataset` uses `np.memmap`. Batches are sampled with random offsets.
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Implement `scripts/prepare_data.py`** and check it by hand with `--max-docs 2000` (expect the three files created; print token count).
- [ ] **Step 6: Commit** `git commit -am "feat: BPE tokenizer and token bin dataset"`

### Task 4: Trainer

**Files:**
- Create: `microllm/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `Config`, `MicroLLM`, `BinDataset`, `Tok`, `pick_device`.
- Produces: `python -m microllm.train` with flags `--data-dir data --out checkpoints/micro-llm.pt --max-steps 20000 --batch-size 32 --grad-accum 4 --lr 6e-4 --warmup 200 --eval-every 500 --eval-batches 20 --resume --compile --require-gpu --seed 1337`.
- Produces: `train(args) -> dict` returning `{"final_val_loss": float, "step": int}`.
- Produces: checkpoint dict keys: `config`, `model`, `optimizer`, `scaler`, `step`, `val_loss`, `tokenizer_sha256`; saved to `--out` (best val) and `--out` with suffix `.last.pt` (latest, used by `--resume`).
- Logs a line per `--eval-every` steps: step, train loss, val loss, tokens/sec, peak VRAM (when on GPU).

- [ ] **Step 1: Write failing tests** (tiny config, synthetic 5k-token bin on CPU): `test_loss_decreases` (train 60 steps on a repeating sequence; final train loss < initial by at least 30%), `test_resume_continues_step` (train 10 steps, resume with `--max-steps 20`; resulting `step == 20` and the optimizer state is loaded), `test_require_gpu_fails_on_cpu_box` (monkeypatch no GPU; expects `RuntimeError`), `test_nonfinite_loss_aborts` (monkeypatch the model to return `nan` loss; trainer raises `RuntimeError` mentioning "non-finite" after 20 consecutive bad steps), `test_checkpoint_loads_on_cpu` (`torch.load(..., map_location="cpu")` has the keys listed above).
- [ ] **Step 2: Run** `pytest tests/test_train.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement the trainer.** AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), linear warmup then cosine decay to 10% of peak, grad clip 1.0, autocast fp16 only when the device is `"cuda"`, `GradScaler` enabled only then. Count consecutive non-finite losses and abort at 20.
- [ ] **Step 4: Run** `pytest tests/test_train.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: resumable fp16 trainer"`

### Task 5: Generate and export

**Files:**
- Create: `microllm/generate.py`, `microllm/export.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: checkpoint format from Task 4, `Tok`, `generate()` from Task 2.
- Produces: `load_checkpoint(path: str, device: str) -> tuple[MicroLLM, Config]`.
- Produces: `python -m microllm.generate --model checkpoints/micro-llm.pt --tokenizer data/tokenizer.json --prompt "Once upon a time" --tokens 200 --temp 0.8 --top-k 40 --seed N`.
- Produces: `python -m microllm.export --model checkpoints/micro-llm.pt --out checkpoints/micro-llm.safetensors` writes weights plus a JSON sidecar with the config.

- [ ] **Step 1: Write failing tests** (tiny random-weight checkpoint saved to `tmp_path`): `test_generate_is_deterministic_with_seed`, `test_prompt_longer_than_context_does_not_crash` (prompt of `block_size + 50` tokens), `test_empty_prompt_uses_eot`, `test_export_roundtrip` (safetensors weights equal checkpoint weights, config JSON equals checkpoint config), `test_load_checkpoint_on_cpu`.
- [ ] **Step 2: Run** `pytest tests/test_generate.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the two modules. The tied head weight is saved once in safetensors (drop `head.weight`, reconstruct on load).
- [ ] **Step 4: Run** `pytest tests/test_generate.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: generation CLI and safetensors export"`

### Task 6: README runbook and GPU validation (on the GPU machine)

**Files:**
- Create: `README.md`, `tests/test_gpu.py`

**Interfaces:**
- Produces: `tests/test_gpu.py::test_forward_backward_on_gpu` marked `@pytest.mark.gpu`: a tiny model's forward and backward pass on the device is finite and the grads are non-zero. Skipped when no GPU.

- [ ] **Step 1: Write `README.md`** with these sections, using exact commands: prerequisites (Ubuntu or similar, ROCm installed, user in `render` and `video` groups, `rocminfo` shows `gfx1030`); setup (`bash scripts/setup_rocm.sh <rocm-version>`); verify (`python scripts/check_gpu.py`); data (`python scripts/prepare_data.py`); smoke run (`python -m microllm.train --max-steps 200 --eval-every 50 --require-gpu`); full run; resume; generate; export; copy checkpoint to the Mac; troubleshooting (`HSA_OVERRIDE_GFX_VERSION=10.3.0` only if the GPU is not detected; out-of-memory means lower `--batch-size` and raise `--grad-accum`; NaN loss means lower `--lr`).
- [ ] **Step 2: Write `tests/test_gpu.py`** as above. Register the `gpu` marker in `pyproject.toml`.
- [ ] **Step 3: On the GPU machine run** `pytest -v` (all tests, including the gpu one). Expected: all PASS.
- [ ] **Step 4: Run the smoke training** from the README. Expected: loss falls from about 9 toward below 6 within 200 steps; tokens/sec and peak VRAM printed; peak VRAM under 14 GB. Record the numbers in the README.
- [ ] **Step 5: Run the full training**, then generate with prompt `Once upon a time`. Expected: coherent short-story English; final val loss recorded in the README.
- [ ] **Step 6: Copy `checkpoints/micro-llm.pt` and `data/tokenizer.json` to the Mac** and run `python -m microllm.generate`. Expected: text generated on MPS or CPU.
- [ ] **Step 7: Commit** `git add -A && git commit -m "docs: README runbook and GPU validation"`

---

## Self-Review Notes

- Spec coverage: model (T2), tokenizer + TinyStories (T3), ROCm fp16 training (T4), Mac-loadable checkpoint and safetensors (T5), setup script, GPU check, README (T1, T6). `--require-gpu` anti-goal covered in T1 and T4.
- Training time on the 6800 XT is unmeasured. A rough guess is 30–60k tokens/sec, so a 20k-step run of about 65k tokens per step (about 1.3B tokens) could take 6–12 hours. Task 6 measures real throughput, and `--max-steps` can be cut down after the smoke run.
