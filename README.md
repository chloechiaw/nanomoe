# nanoMoE

> nanoMoE: a minimal Mixture-of-Experts language model 


Trained in 5–7 hours on a single A10G (24 GiB) via Modal. At d_model=256 / 8 layers / 8 experts top-2, only 33% of params are active per token (12.6M of 37.8M, ex-embeddings), and it reaches 1B tokens in ~2.2h — well inside budget. 

 `scripts/size_moe.py --sweep` finds configs in the target band, 
 `scripts/bench_step.py` gives you the GPU-side step-time ceiling before you commit to a run, and `--n-expert=1` falls straight back to the dense MLP (upstream's 48 tests still pass unmodified).

### Quantile Balancing (QB)

Instead of an aux load-balancing loss with a coefficient to tune, QB adds a per-expert `router_bias` recomputed in closed form every step. Setting each `-bias[e]` to the fair-share quantile of that expert's logits hands it exactly its share of tokens *by construction*, rather than nudging toward it. **Nothing to tune** — no balance coefficient, no update rate. Balance is reported as `MaxVio = (max_load - mean_load) / mean_load`; 0 is perfect. Ported from [Marin](https://github.com/marin-community/marin)'s JAX implementation, idea from Jianlin Su.

### Usage

A10G Modal instance.

```bash
pip install modal && modal setup                    # once

modal run modal_app.py::probe                        # ~2 min,  is the GPU stack sane?
modal run modal_app.py::prepare --shards 32          # ~30 min, data + tokenizer + eval
modal run modal_app.py::smoke                        # ~10 min, 30 real steps -> tok/sec
```

Train (`--args` is forwarded verbatim to `base_train.py` and overrides the A10G defaults):

```bash
modal run modal_app.py::train --args "--depth=8"
modal run modal_app.py::train --args "--depth=10 --device-batch-size=4 --num-iterations=8000"
```

Runs resume from the last checkpoint automatically:

```bash
modal run modal_app.py::train --args "--resume-from-step=4000"
```

Eval:

```bash
modal run modal_app.py::evaluate --args "--model-tag=d8"
```


- **bf16, no fp8, no FA3** on sm_86 — confirmed, not assumed. Don't "optimise" `--window-pattern=L` back.
- The router's gradient is exactly zero on step 0 (c_proj is zero-init); it starts learning one step later. QB is unaffected since it reads logits, not gradients. `tests/test_moe.py` pins this.
- Batch/LR/weight-decay scaling laws are inherited from dense nanochat and *not* re-derived for MoE — the training script warns you.
