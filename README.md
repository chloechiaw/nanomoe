# nanoMoE

A minimal Mixture-of-Experts language model for fast experimentation! 

## Reproducing the 5 hour run

One H100. About 5 hours of training and $21 on Modal. Params are 498M total and 184M active.

### Setup

```bash
pip install modal && modal setup
export NANOMOE_GPU=H100
export WANDB_API_KEY=...        # optional, for live curves
```

### Data

```bash
modal run modal_app.py::prepare --shards 120 --vocab-size 8192
```

Downloads 120 ClimbMix shards, trains an 8192 token BPE tokenizer.

### Check the GPU before committing

```bash
modal run modal_app.py::smoke        # 30 real steps, prints tokens/sec
```

### Train

```bash
modal run --detach modal_app.py::train --run nanomoe-h100 --args \
  "--depth=16 --aspect-ratio=40 --head-dim=64 --window-pattern=L \
   --n-expert=8 --top-k=2 --moe-dispatch=grouped \
   --device-batch-size=32 --num-iterations=7350 \
   --model-tag=moe-d16-h100 \
   --eval-every=500 --eval-tokens=10485760 \
   --core-metric-every=1500 --core-metric-max-per-task=200 \
   --sample-every=-1 --expert-load-every=200 --save-every=2500"
```

### Resume checkpointing 
Resume from the newest checkpoint after an interruption, passing the same architecture
flags:

```bash
modal run --detach modal_app.py::train --run nanomoe-h100 --resume --args "<same flags>"
```

### Evaluate

```bash
modal run --detach modal_app.py::evaluate --args "--model-tag=moe-d16-h100 --device-batch-size=8"
modal run --detach modal_app.py::mmlu --args "--model-tag=moe-d16-h100"
```

`evaluate` writes per task accuracy to
`nanochat/base_eval/base_model_007350.csv` on the volume. Pull it down with:

```bash
modal volume get nano-moe-data nanochat/base_eval/base_model_007350.csv .
```
