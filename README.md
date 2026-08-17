# nanoMoE

A toy Mixture-of-Experts for fast experimentation. We utilize nanochat but with an MoE block that replaces the transformer block's dense MLP. 

## Training 
You can train a reasonably performing Mixture of Experts model for <$20 on 1x H100 using 3.5-4 less OOMs than the smallest open source MoEs (480x below Pythia-1B's 10^2.7 FLOPs and 9,500x below OLMoE-1B-7B's 10^4.0 FLOPs). 6ND arithmetic used for FLOPs during training

<img width="2600" height="940" alt="nanomoe_frontier" src="https://github.com/user-attachments/assets/34400ee1-1052-41c6-997c-888b22c1fee6" />

### Setup

```bash
pip install modal && modal setup
export NANOMOE_GPU=H100
export WANDB_API_KEY=...       
```

### Data

```bash
modal run modal_app.py::prepare --shards 120 --vocab-size 8192
```

Downloads 120 ClimbMix shards, trains an 8192 token BPE tokenizer.

### Train

`TRAIN_DEFAULTS` in `modal_app.py` is exactly the config that produced the numbers above, so
this reproduces the run:

```bash
modal run --detach modal_app.py::train --run nanomoe-h100
```

Alternatively: 

```bash
modal run --detach modal_app.py::train --run nanomoe-h100 --args \
  "--depth=16 --aspect-ratio=40 --head-dim=64 --window-pattern=L \
   --n-expert=8 --top-k=2 \
   --device-batch-size=32 --num-iterations=7350 \
   --eval-every=500 --eval-tokens=10485760 \
   --core-metric-every=1500 --core-metric-max-per-task=200 \
   --expert-load-every=200 --save-every=2500"
```

### Evaluate

```bash
modal run --detach modal_app.py::evaluate --args "--device-batch-size=8"
```

### Evaluate

Scores ARC-Easy, PIQA and HellaSwag on the full test sets, plus train/val bits-per-byte.

```bash
modal run --detach modal_app.py::evaluate --args "--device-batch-size=8"
```
### Experiments / sweeps

Each `modal run --detach` grabs its own H100, so a sweep is just a loop. The MoE knob worth
sweeping is granularity: hold `top_k / n_expert` fixed so active FLOPs per token stay put,
and vary how finely the same compute is split. `sweep.sh`:

```bash
--eval core                  # benchmarks only, skip bpb
--eval bpb                   # bpb only, ~1 min
--max-per-task=200           # sample instead of the full test set, for a quick read
--step=2500                  # an earlier checkpoint (default: the newest)
--model-tag=<name>           # a specific run's checkpoint dir (default: `d<depth>`)
```

Checkpoints go to `d<depth>/` on the volume unless you pass `--model-tag`. Runs at the same
depth therefore overwrite each other, so give each point in a sweep its own tag (on both the
train and evaluate commands).

Results land on the volume as `base_eval/base_model_<step>.csv`, one row per benchmark with
raw and chance-centered accuracy. To find and fetch them:

for c in "${configs[@]}"; do
    read -r e k <<< "$c"
    modal run --detach modal_app.py::train --run "gran-e$e-k$k" --args \
      "--depth=8 --aspect-ratio=40 --n-expert=$e --top-k=$k \
       --target-flops=5e17 --model-tag=gran-e$e-k$k"
done
```

### Pythia and OLMo comparisons

nanoMoE reaches 97% of Pythia-1B's average across these six benchmarks and 87% of OLMo-1B's,
with 6-7x fewer active parameters.

| benchmark | chance | nanoMoE | Pythia-1B | % of | OLMo-1B | % of |
|---|---|---|---|---|---|---|
| MMLU | 25 | 26.0 | 31.1 | 84% | 32.1 | 81% |
| HellaSwag | 25 | 43.8 | 48.0 | 91% | 67.5 | 65% |
| ARC-Challenge | 25 | 33.2 | 31.4 | **106%** | 36.4 | 91% |
| ARC-Easy | 25 | 57.7 | 63.4 | 91% | 53.5 | **108%** |
| PIQA | 50 | 70.2 | 68.9 | **102%** | 74.0 | 95% |
| WinoGrande | 50 | 54.4 | 52.7 | **103%** | 62.9 | 86% |
| **average** | | **47.5** | **49.3** | **97%** | **54.4** | **87%** |

| | active params | total params |
|---|---|---|
| nanoMoE | 0.18B | 0.50B |
| Pythia-1B | 1.1B (dense) | 1.1B |
| OLMo-1B | 1.3B (dense) | 1.3B |

