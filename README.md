# nanoMoE

A toy Mixture-of-Experts language model for fast experimentation! 

## Training 
Using this repo, you can train a reasonable performing Mixture of Experts model for <$20 on an H100. 
* 4.12e18 FLOPs
* 3.85B tokens in 5.1 hours on a single H100, much less compute than the smallest MoEs out there (480x less than Pythia-1B and 9,500x less than OLMoE-1B-7B). 

<img width="1289" height="462" alt="png" src="https://github.com/user-attachments/assets/ce1adbb7-da66-4860-8d6f-f65eb04b5549" />

I referred to the [OLMoE paper](https://arxiv.org/abs/2409.02060) where they have a table of varying MoE sizes and their performance on 8 benchmarks. The smallest band is 1B active parameters (compare this with nanochat's 561M params), which means a routing experiment costs cluster time and days.The architecture follows a typical MoE, the only new thing I added was * [quantile balancing (from Jianlin Su, used in Kimi K3]. This is great because we don't need to do hyperparameter sweeps and also deals with load balancing. 

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

### Todo:

More ablations 
