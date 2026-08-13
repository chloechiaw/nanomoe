# nanoMoE

A toy Mixture-of-Experts for fast experimentation. We utilize nanochat but with an MoE block that replaces the transformer block's dense MLP. 

## Training 
You can train a reasonable performing Mixture of Experts model for <$20 on 1x H100 using 3-4 less OOMs. 
* 4.12e18 FLOPs
* 3.85B tokens in 5.1 hours

480x below Pythia-1B (10^2.7) and 9,500x below OLMoE-1B-7B (10^4.0)

<img width="1289" height="462" alt="png" src="https://github.com/user-attachments/assets/ce1adbb7-da66-4860-8d6f-f65eb04b5549" />

I referred to the [OLMoE paper](https://arxiv.org/abs/2409.02060) where they have a table of varying MoE sizes and their performance on 8 benchmarks. The smallest category of MoEs they use is 1B active parameters (compare this with nanochat's 561M params), which means a routing experiment costs cluster time and days.The architecture follows a typical MoE, the only new thing I added was quantile balancing (from Jianlin Su, used in Kimi K3). This is great because we don't need to do (hyperparameter sweeps)[https://openathena.ai/blog/quantile-balancing/] and also deals with load balancing. 

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
modal run --detach modal_app.py::train --run nanomoe-h100 --args "--model-tag=moe-d16-h100"
```

Spelled out, in case you want to change something:

```bash
modal run --detach modal_app.py::train --run nanomoe-h100 --args \
  "--depth=16 --aspect-ratio=40 --head-dim=64 --window-pattern=L \
   --n-expert=8 --top-k=2 \
   --device-batch-size=32 --num-iterations=7350 \
   --model-tag=moe-d16-h100 \
   --eval-every=500 --eval-tokens=10485760 \
   --core-metric-every=1500 --core-metric-max-per-task=200 \
   --expert-load-every=200 --save-every=2500"
```

### Resume checkpointing 
Resume from the newest checkpoint after an interruption, passing the same architecture
flags:

```bash
modal run --detach modal_app.py::train --run nanomoe-h100 --resume --args "<same flags>"
```

### Evaluate

Scores ARC-Easy, PIQA and HellaSwag on the full test sets, plus train/val bits-per-byte.
Point it at any checkpoint with `--model-tag`:

```bash
TAG=moe-d16-h100
modal run --detach modal_app.py::evaluate --args "--model-tag=$TAG --device-batch-size=8"
```

Useful variations:

```bash
--eval core                  # benchmarks only, skip bpb
--eval bpb                   # bpb only, ~1 min
--max-per-task=200           # sample instead of the full test set, for a quick read
--step=2500                  # an earlier checkpoint (default: the newest)
```

Results land on the volume as `base_eval/base_model_<step>.csv`, one row per benchmark with
raw and chance-centered accuracy. To find and fetch them:

```bash
modal run modal_app.py::ls --path base_eval
modal volume get nano-moe-data nanochat/base_eval/base_model_<step>.csv .
```

Read the centered column, not the raw one. PIQA starts at 50% and the other two at 25%, so
raw accuracy overstates how much the model actually knows.

### Todo:

More ablations 
