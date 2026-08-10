# nanoMoE

A small, legible Mixture-of-Experts language model: **fewer active parameters, performance
consistent with the published MoE frontier**, trained in 5–7 hours on a single GPU.

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat) at
[`92d63d4`](https://github.com/karpathy/nanochat/commit/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd)
(2026-07-03). The plan, the claims it does and does not make, and the deliberate omissions are
in [BRIEF.md](BRIEF.md).

## Status

| | |
|---|---|
| Base harness vendored from nanochat | done |
| Modal / A10G training harness | done — `modal_app.py` |
| Config sizing tool | done — `scripts/size_moe.py` |
| MoE FFN block (8 experts, top-2) | done — `MoEMLP` in `nanochat/gpt.py` |
| Quantile Balancing (QB) load balancing | done — same file, `apply_qb_update` |
| Step-time benchmark vs. the budget | done — `scripts/bench_step.py` |
| Full training run | not started |
| MoE-vs-MoE frontier plot | not started |

`--n-expert=1` keeps the dense MLP, so upstream nanochat is still reachable and its test suite
(48 tests) still passes unmodified.

## Running on Modal

Everything runs against one **A10G** (24 GB, GA102/sm_86). State lives on a Modal Volume
(`nano-moe-data`) mounted at `/data`, so the expensive prepare steps happen once and a killed
run keeps its checkpoints.

```bash
pip install modal && modal setup       # once

modal run modal_app.py::probe                       # ~2 min,  is the GPU stack sane?
modal run modal_app.py::prepare --shards 32         # ~30 min, data + tokenizer + eval bundle
modal run modal_app.py::smoke                       # ~10 min, 30 real steps -> tok/sec
modal run modal_app.py::train --args "--depth=8"    # the real run
```

`--args` is forwarded verbatim to `scripts/base_train.py` and overrides the A10G defaults in
`TRAIN_DEFAULTS` (argparse keeps the last occurrence of a flag), so:

```bash
modal run modal_app.py::train --args "--depth=10 --device-batch-size=4 --num-iterations=8000"
modal run modal_app.py::train --run my-run-name     # logs to wandb if WANDB_API_KEY is exported
modal run modal_app.py::evaluate --args "--model-tag=d8"
```

Checkpoints:

```bash
modal volume ls  nano-moe-data nanochat/base_checkpoints
modal volume get nano-moe-data nanochat/base_checkpoints/d8 ./checkpoints
```

Interrupted runs resume from the last saved step (`--save-every` defaults to 1000, and the
harness commits the volume every 10 minutes regardless):

```bash
modal run modal_app.py::train --args "--resume-from-step=4000"
```

## What the hardware forces

Measured on Modal, `2026-08-09`, via `modal run modal_app.py::probe`:

```
GPU              : NVIDIA A10  (sm_86, 22.1 GiB)
torch            : 2.9.1+cu128  cuda 12.8
compute dtype    : torch.bfloat16 (auto-detected: CUDA SM 86 (bf16 supported))
FA3 available    : False
FA3 in use       : False
measured 8k bf16 matmul: 7.939e+13 FLOP/s
```

- **Modal's `A10G` tier scheduled a plain A10** (72 SM, not the A10G's 80). Treat 125 TFLOPS
  dense bf16 as the peak and re-run `probe` if throughput looks off — it prints what you got.
- **No Flash Attention 3 — confirmed, not assumed.** nanochat pulls FA3 from the HF `kernels`
  hub and its loader claims sm_80/sm_86 coverage, but no build is actually published for sm_86,
  so it falls back to PyTorch SDPA. **This is why `--window-pattern=L` is the default**: SDPA
  has no fused sliding-window support, so the upstream `SSSL` pattern would materialise an
  explicit `T×T` bool mask per short-window layer. Do not "optimise" that flag back.
- **bf16, no fp8.** sm_86 has no FP8 tensor cores, so `--fp8` is off. nanochat's dtype
  autodetect picks bf16 (SM ≥ 8.0) on its own.
- **~125 TFLOPS peak** is ~1/8th of one H100, so the 5–7 hour budget buys roughly what an
  8×H100 speedrun gets through in a couple of minutes.
- **Single GPU, no DDP.** `world_size=1`, so `total_batch_size` is reached purely by gradient
  accumulation over `device_batch_size`.
- **24 GB, and the logit tensor is what fills it.** `base_train` materialises `B*T*vocab` in
  bf16, again in fp32, again after the tanh softcap, and again inside `cross_entropy`. At B=8,
  T=2048, vocab=32768 that chain alone is ~7.5 GiB. Default is `--device-batch-size=8`; drop to
  4 if you OOM. MoE will add pressure from the other side: 8 experts means 8× the FFN weights
  and 8× their optimizer state resident, even though only top-2 are active per token.

## The MoE block

`MoEMLP` in [nanochat/gpt.py](nanochat/gpt.py) replaces `MLP` when `--n-expert > 1`. Each expert
*is* an unmodified nanochat `MLP` (`c_fc` → relu² → `c_proj`), so the structural diff really is
"one block": a router, N copies, and a top-k dispatch.

Two implementation choices that are load-bearing:

- **Every expert runs every micro-batch, even with zero tokens routed to it.** There is no
  `if tok.numel() == 0: continue`. Muon does `torch.stack([p.grad for p in group])` over its
  parameter groups, and a starved expert would contribute `None` and crash the optimizer step.
  An empty matmul yields exactly the zero gradient that keeps the stack well-formed.
- **The dispatch loop is `@torch._dynamo.disable`d.** `torch.where` has a data-dependent output
  shape, so under `torch.compile` this region either graph-breaks anyway or recompiles every
  step as per-expert token counts drift. Every tensor crossing the boundary is statically
  shaped, so the rest of the model still compiles.

The dispatch is the naive gather/scatter loop the brief calls for. It is genuinely **dropless**
— no capacity factor, no token dropping — and does exactly `top_k` experts' worth of FLOPs, not
`n_expert`'s. The faster path is stacked expert weights plus a grouped matmul; see the measured
MoE-vs-dense ratio below before reaching for it.

### Quantile Balancing

QB replaces the aux loss. Routing picks the top-k of `router_logits + router_bias`, and the bias
is recomputed in closed form each step — no balance coefficient, no update rate, nothing to
tune, which is exactly why the brief chose it (it removes a confound from the comparison).

1. Per token, the selection threshold `alpha_t` is the **(k+1)-th** largest biased logit — which
   is why the code takes `topk(k+1)` and throws the last index away.
2. Expert `e` is selected iff `logits[t,e] - alpha_t > -bias[e]`. So setting `-bias[e]` to the
   `(N*k/E)`-th largest value of `logits[:,e] - alpha` hands `e` exactly its fair share *by
   construction*, rather than nudging toward it.

Betas are accumulated across the gradient-accumulation micro-steps of one optimizer step (the
single-GPU stand-in for the reference's `pmean` over data-parallel shards) and applied at the
next step, matching the reference's pipelining. Balance is reported as **MaxVio**
`(max_load - mean_load) / mean_load`, logged every `--expert-load-every` steps; 0 is perfect.

Ported from Marin's JAX implementation
([model.py](https://github.com/marin-community/marin/blob/c4ce3ae9e427e57d625ece10248911c5310e5991/experiments/grug/moe/model.py#L349-L394),
[train.py](https://github.com/marin-community/marin/blob/c4ce3ae9e427e57d625ece10248911c5310e5991/experiments/grug/moe/train.py#L240-L253)),
original idea from [Jianlin Su](https://kexue.fm/archives/11619).

One quirk worth knowing: **the router's gradient is exactly zero on step 0.** Its only gradient
path is the sigmoid combine weight, which multiplies the expert output — and nanochat zero-inits
every `c_proj`, so that output starts at zero. The router begins learning one optimizer step
later. QB is unaffected, since it reads logits rather than gradients, so load balancing is live
from step 0. `tests/test_moe.py` pins this behaviour so it can't silently become a real bug.

## Measured step time vs. the budget

`modal run modal_app.py::bench`, on the A10 described above. Random tokens, no dataloader, so
these are the GPU-side ceiling. `d_model=256, n_layer=8, seq 2048`, 65,536 tokens/optimizer step.

| config | vocab | dev batch | step | tok/s | MFU | peak mem | 1B tokens |
|---|---|---|---|---|---|---|---|
| MoE 8x top-2 | 32768 | 8 | 892 ms | 73.5k | 9.6% | 11.11 GiB | **3.78 h** |
| MoE 8x top-2 | 8192 | 8 | 514 ms | 127.5k | 12.8% | 4.78 GiB | **2.18 h** |
| MoE 8x top-2 | 8192 | 32 | 412 ms | 159.2k | 16.0% | 17.05 GiB | **1.74 h** |
| dense (`--n-expert=1`) | 8192 | 8 | 159 ms | 411.2k | 33.1% | 1.66 GiB | 0.68 h |

**The budget is met with room to spare.** The target MoE reaches 1B tokens in 1.7–3.8 h against
a 5–7 h window, so the naive dispatch does not need replacing.

Four things worth reading off this table:

- **MoE costs 3.2x dense wall-clock** (514 / 159 ms). The brief's rule of thumb was "fine at
  2–3x, reconsider at 10x+", so this sits just at the top of the acceptable band and a
  ragged/grouped-matmul kernel stays deferred. Note only **1.25x** of that 3.2x is real extra
  work (1.259e8 vs 1.007e8 FLOPs/token); the other ~2.6x is dispatch overhead — eight gather →
  small-matmul → scatter round trips per layer, in eager mode. That is the headroom a fast
  kernel would recover, and it is the honest reason MFU drops from 33.1% to 12.8%.
- **Vocab 8192 is 1.74x faster and uses 2.3x less memory** than 32768, on an identical trunk.
  This is a much larger effect than the parameter-accounting argument, and it points the same
  way. At 4.78 GiB of 22 GiB there is a lot of room to raise `--device-batch-size`.
- **Raising `--device-batch-size` from 8 to 32 buys 25% throughput** (127.5k → 159.2k tok/s) at
  identical tokens-per-optimizer-step, which is the dispatch overhead amortising over larger
  per-expert matmuls. `--device-batch-size` is only a memory/throughput knob: `base_train.py`
  computes `total_batch_size` independently and reaches it via gradient accumulation, so
  changing it does not change the training math.
  **32 is the measured ceiling** at vocab 8192 — 40, 48 and 64 all OOM (40 asks for 2.50 GiB
  more with 640 MiB free). At vocab 32768 the ceiling is far lower, since B=8 already costs
  11.11 GiB. Past 32 the remaining throughput is in the dispatch kernel, not the batch size.
- **Dense hits 33.1% MFU**, which independently corroborates the 125 TFLOPS peak derived in
  `get_peak_flops` — a plausible cuBLAS-limited number rather than the >100% the datasheet
  figure produced.

The wall-clock headroom is real but should not be read as "we can train 3x longer for free":
these runs use random tokens, so the real run additionally pays CPU-side BPE tokenization in
the dataloader. Confirm with `smoke` once `prepare` has run.

## Sizing a config

```bash
python -m scripts.size_moe --sweep                            # find configs in the target band
python -m scripts.size_moe --d-model 256 --n-layer 8 --vocab-size 8192
```

Two things fall straight out of this, both of which need deciding **before** the tokenizer step:

**1. Vocab size dominates at this scale.** At nanochat's default 32768 vocab and d_model=256,
`wte + value_embeds + lm_head` is ~50M params against a ~44M MoE trunk — the embeddings are
bigger than the thing being measured, and the "active parameter" headline becomes mostly an
embedding-table statistic. `value_embeds` alone (nanochat puts a `padded_vocab × kv_dim`
embedding on every other layer) is the single largest block. Dropping to `--vocab-size=8192`
puts embeddings at ~12.6M against a 37.8M trunk, and `val_bpb` is vocab-invariant so the loss
metric stays comparable either way.

**2. At the target size the GPU is probably not the bottleneck.** d_model=256 / 8 layers /
8 experts top-2 is ~1.26e8 FLOPs/token, which even at a modest 30% MFU is ~6B tokens in 6 hours
— roughly 10× the brief's 500M–1B budget. The binding constraint is more likely the dataloader,
which BPE-tokenizes on CPU inside the training process, or the naive masked-loop MoE once it
lands. Read the actual `tok/sec` off `smoke` before believing any of this; that is what `smoke`
is for. If the GPU really is idle, the honest response is to spend the headroom on a bigger
model or more tokens rather than to report a 6-hour number that was never compute-bound.

For reference, the sizing tool's output for the leading candidate:

```
d_model=256  n_layer=8  n_head=4  d_ff=1024  experts=8 top-2  vocab=8192
    total  ex-embeddings     37.77M
    active ex-embeddings     12.60M   -> active frac 33.4%
    FLOPs/token          1.259e+08
```

## Layout

```
modal_app.py          Modal harness: image, volume, prepare/train/eval/probe functions
BRIEF.md              the plan, the claims, and the explicit non-goals
scripts/size_moe.py   param & FLOP accounting for candidate configs
scripts/base_train.py pretraining loop            (nanochat, unmodified)
scripts/base_eval.py  CORE + bpb + samples        (nanochat, unmodified)
nanochat/gpt.py       the transformer — the MoE FFN swap goes here
nanochat/optim.py     Muon + AdamW                (nanochat, unmodified)
nanochat/dataloader.py BOS-aligned best-fit packing (nanochat, unmodified)
```

### Local diff against upstream nanochat

- `nanochat/common.py` — added A10G / A10 entries to `get_peak_flops` and `get_peak_bandwidth`,
  so MFU and MBU report real numbers instead of 0%. (A10G before A10: `"a10"` is a substring of
  `"NVIDIA A10G"` and would otherwise match first.) The FLOPS values are computed from
  `SMs × boost_clock × 1024`, not copied from NVIDIA's datasheets, which quote the two parts
  inconsistently — the A10G sheet's 70/140 is a dense/sparse pair while the A10 sheet's 125 is
  sparse-only. Taking the A10 sheet at face value gave 62.5 TFLOPS and a 127% MFU reading.
- `scripts/size_moe.py`, `modal_app.py`, `BRIEF.md`, this README — new.

Everything else is upstream and unmodified. Keeping it that way is the point: the MoE story is
"change exactly one block", and that claim is only legible if the diff stays small.

## Known limitations

Stated up front rather than buried, per BRIEF.md:

- **The batch-size / LR / weight-decay scaling laws are inherited from dense nanochat and have
  not been re-derived for MoE.** `base_train.py` picks the token horizon, batch size, LR
  correction and weight-decay scaling from fits made on dense models, using a dense d12
  reference. Whether `B_opt ∝ D^0.383` and `η ∝ √(B/B_ref)` transfer to a sparse model is an
  open question; the training script prints a warning saying so. Re-deriving them would cost
  more compute than the whole run.
- No LR sweep, single seed, no dense baseline run — all deliberate, see BRIEF.md.
- The router is optimized by Muon along with every other matrix in `transformer.h`, because
  that is what upstream does with anything under that module. Most MoE implementations use
  AdamW for the router. Untested here either way.
- MaxVio is measured on the *training* batches only; `model.eval()` suppresses accumulation.

## Next

1. Decide vocab size (the benchmark above argues hard for 8192), then run `prepare`.
2. `smoke` for the end-to-end tok/sec including the dataloader, and a first look at MaxVio.
3. Full training run; watch MaxVio and the per-expert share in the logs.
4. Build the MoE-vs-MoE frontier plot and the expert-load-over-training figure.
