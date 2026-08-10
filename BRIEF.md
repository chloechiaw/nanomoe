# nanoMoE — Project Brief

## Goal
Build a small, legible MoE language model (forked from/inspired by karpathy's `nanochat`) that
demonstrates: **fewer active parameters, comparable performance to other MoEs**, trainable in
5–7 hours on a single GPU. This is a portfolio/demo project aimed at showing ML-systems and
MoE expertise to MatX (a company with a strong Rust/performance-systems angle) — but the
project itself is written in **PyTorch**, not Rust or JAX. No browser/WASM component. No live
in-browser training.

## Base codebase
Fork/extend `karpathy/nanochat` (https://github.com/karpathy/nanochat):
- Single-GPU-friendly, minimal, readable PyTorch training harness.
- Has its own `val_bpb` (bits-per-byte, vocab-invariant) loss metric and CORE eval script.
- File structure to reuse: `nanochat/gpt.py` (transformer), `nanochat/optim.py` (AdamW + Muon),
  `nanochat/dataloader.py`, `scripts/base_train.py`, `scripts/base_eval.py`.

## Architecture: minimal dense → MoE switch
Keep everything else in nanochat identical (attention, tokenizer, training loop, data). Change
exactly one block:
- **Dense FFN**: `Linear(d_model, d_ff) → activation → Linear(d_ff, d_model)`
- **MoE FFN**: same up/down projection shape, replicated across N experts + a router
  (`Linear(d_model, N)`), top-k experts selected per token, weighted-sum outputs.
- **Config**: 8 experts total, top-2 active per token (Mixtral-style ratio — standard,
  well-precedented, avoids being an arbitrary config choice).
- Target sizing: ~30–60M total params, ~10–20M active params (active fraction ~30–40%).
- Implementation can start **naive** (loop over experts, mask non-selected tokens) — correct
  but slower. Only add a dropless kernel (ragged_dot-style) if naive step time eats too far into
  the token budget (see "Dropless kernels" section below).

## Load balancing: Quantile Balancing (QB)
Use **Quantile Balancing** instead of aux-loss / aux-loss-free bias — it's hyperparameter-free,
which removes a confound from the isoFLOP-style comparison (no balance-loss coefficient to tune
per model size).
- Source: Jianlin Su, Feb 2026 blog post (https://kexue.fm/archives/11619).
- Validated at scale by Marin (32B-A5B, 1e22 FLOPs, 326B tokens) — zero loss spikes, no need for
  leading dense layers or aux losses. Write-up: https://openathena.ai/blog/quantile-balancing/
- Reference JAX implementation (needs porting to PyTorch):
  - MoE forward (bias application + QB-β computation):
    https://github.com/marin-community/marin/blob/c4ce3ae9e427e57d625ece10248911c5310e5991/experiments/grug/moe/model.py#L349-L394
  - `_apply_qb_betas` helper:
    https://github.com/marin-community/marin/blob/c4ce3ae9e427e57d625ece10248911c5310e5991/experiments/grug/moe/train.py#L240-L253
- Mechanism (2-step, per training step, based on router scores):
  1. For each token, compute the threshold score required to become activated.
  2. For each expert, using that threshold, determine the bias that would activate a balanced
     number of tokens.
- Nice side benefit: produces a legible "per-expert token count over training" plot for the
  README, mirroring Marin's own published QB visualizations.

## Training budget (5–7 hours, single GPU)
- Token budget: fixed, matched across any variants you run — 500M–1B tokens is a realistic
  target for a ~40M-param model in this time window.
- **Skip for this pass** (explicitly documented as known limitations, not hidden gaps):
  - No LR sweep per config — pick one reasonable LR, document it.
  - No multiple seeds.
  - No dense baseline runs (comparison is MoE-vs-MoE only, per latest decision).
- Primary comparison is against **published external MoE models**, not a self-trained dense
  baseline — this avoids burning training budget on a baseline that isn't the point.

## Evaluation plot: MoE vs. MoE frontier (NanoVQA-style)
Modeled on ellenjxu/nanovqa's Pareto frontier plot
(https://github.com/ellenjxu/nanovqa) — small "NanoX" star at bottom-left, larger published
models plotted as reference dots, connected by a dashed pareto frontier line.

- **X-axis**: active parameters (or active-FLOPs), log scale.
- **Y-axis**: MMLU (most universally reported benchmark across all reference models below).
- **Reference points** (real, citable numbers):
  - **OLMoE-1B-7B**: 6.9B total, 1.3B active, pretrained on 5.1T tokens. Best-in-class under 2B
    active params; fully open (weights, data, code, logs). Source: arXiv 2409.02060.
  - **Qwen1.5-MoE-A2.7B**: 2.7B active, matches 7B-class dense models (Mistral-7B, Qwen1.5-7B).
    4 shared experts always active + 60 routed experts, top-4 selected. Source: Qwen blog.
  - **DeepSeek-V2-Lite**: 15.7B total, 2.4B active, MLA attention, 27 layers, 2 shared experts +
    64 routed experts, top-6. Source: arXiv (DeepSeek-V2 paper family).
  - **JetMoE-8B**: ~2B active. Reference point, generally outperformed by OLMoE in Ai2's own
    benchmarking.
- **Your star**: toy MoE, far bottom-left (tens of millions active params vs. billions for the
  reference models — expect near-chance MMLU, same honest framing as NanoVQA's own star sitting
  well below the bigger VLMs).
- **Honest framing for the README**: this plot does NOT claim to beat OLMoE/Qwen1.5-MoE on
  absolute accuracy — it's not plausible at this compute budget and shouldn't be implied. The
  legitimate claim is: does the toy model's point sit roughly on the extrapolated trend line
  formed by the larger open MoEs, i.e., is its efficiency *consistent with* the frontier these
  bigger models define, even at drastically smaller scale.

## Data efficiency angle (optional, if time allows)
- MoEs are generally *more* data-hungry per-active-param at small scale than dense — each expert
  individually gets fewer gradient updates per token processed (since only a subset of experts
  see any given token). Worth measuring directly: does your MoE need more tokens than a
  comparable dense model to reach the same loss, and at what token count (if any) does it cross
  over to being more efficient? This is a more honest/nuanced finding than a blanket "MoE = less
  data" claim, which isn't well-supported in the literature.
- Fine-grained experts (more, smaller experts, more of them active — DeepSeek-V2/V3's approach)
  is the main lever for improving the compute/data efficiency frontier, if pursued as a stretch
  goal.

## Explicitly deferred / out of scope for this pass
- Shared experts (DeepSeek-style) — good "future work" mention, not required for the core story.
- Multi-token prediction (MTP) — orthogonal efficiency lever, mention only as future work.
- DeepSeek Sparse Attention (DSA) — different axis (attention vs. FFN sparsity), future work.
- Dropless kernels (ragged_dot / MegaBlocks-style) — only add if naive masked-loop MoE step time
  is eating too far into the 5–7 hour token budget (rough rule of thumb: fine if naive step time
  is ~2–3x dense step time; reconsider if it's 10x+). Not required for the loss/accuracy claims,
  only for wall-clock feasibility.
- No seqax-style JAX tooling — this project is PyTorch only.
- No Rust/WASM component for this specific project (kept separate from the earlier
  browser-demo brainstorm for MatX).

## Compute: Modal, single A10-class GPU
Training runs on [Modal](https://modal.com) against a single GA102 / sm_86 card (24 GB). Modal's
`A10G` tier has been observed to schedule a plain **A10** (72 SM). See `README.md` for the run
commands and `modal_app.py` for the harness.

Consequences of this choice, all of which are already handled in the harness:
- **bf16, not fp8.** sm_86 has no FP8 tensor cores, so `--fp8` is off. nanochat's dtype
  autodetect picks bf16 (SM ≥ 8.0), which is what we want.
- **No Flash Attention 3** — verified, not assumed: no sm_86 build is published on the HF
  `kernels` hub, so nanochat falls back to PyTorch SDPA. Since SDPA has no fused sliding-window
  support, `--window-pattern=L` (full context every layer) is mandatory; the upstream `SSSL`
  default would build an explicit T×T mask per short-window layer.
- **Peak bf16 ≈ 125 TFLOPS dense** (an 8192³ GEMM measures 79 TFLOPS, ~63% of that). This is
  ~1/8th of one H100, so the 5–7 hour budget buys roughly the tokens an 8xH100 speedrun gets
  through in a couple of minutes — the model has to be sized accordingly.
- **Single GPU, no DDP.** `world_size=1`, so `total_batch_size` is reached purely by gradient
  accumulation over `device_batch_size`.
- **24 GB VRAM** is the binding constraint on `device_batch_size`, and MoE makes it worse: 8
  experts means 8x the FFN weights and 8x their optimizer state resident, even though only
  top-2 are active per token.
- Volume-backed state (`/data`) so data, tokenizer, and checkpoints survive across runs and a
  killed job can resume with `--resume-from-step`.

## Open questions to resolve when building
- Exact model dims (d_model, n_layer, n_head, d_ff per expert) to hit ~40M total / ~15M active
  params target. Use `scripts/size_moe.py` to explore the space.
- **Vocab size.** At a ~40M-param target, nanochat's default 32768 vocab is a problem: `wte` +
  `lm_head` alone are `2 * 32768 * d_model`, which at d_model=256 is 16.8M params — comparable
  to the entire MoE trunk we're trying to measure. Either (a) retrain the tokenizer at a smaller
  vocab (`scripts.tok_train --vocab-size=8192`), or (b) report params excluding embeddings
  (Kaplan convention) and say so explicitly. `val_bpb` is vocab-invariant either way, so the
  loss metric stays comparable across the choice. This has to be settled *before* the tokenizer
  step, since everything downstream depends on it.
- Whether to also pull a FLOPs-axis version of the reference-model plot (total training FLOPs
  for OLMoE/Qwen1.5-MoE/DeepSeek-V2-Lite) as an alternative/companion to the active-params axis.
- Final corpus choice for training (small FineWeb/OpenWebText-style subset, matched to whatever
  nanochat's dataloader already supports). Currently: ClimbMix-400B, nanochat's own default.
