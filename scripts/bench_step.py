"""
Time the training step and project it against the wall-clock budget.

Uses random tokens rather than the real dataloader, deliberately: this isolates the GPU-side
cost of the model so that a slow number means the model, not CPU-side BPE tokenization. Run
`scripts.base_train` for the end-to-end figure.

    python -m scripts.bench_step --depth 8 --n-expert 8 --top-k 2
    python -m scripts.bench_step --depth 8 --n-expert 1            # dense, for the ratio

Reports ms/step, tok/sec, MFU and peak memory, then the tokens that buys in the target budget.
"""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_init, get_peak_flops, print0
from nanochat.gpt import GPT, GPTConfig


def build(args, device):
    model_dim = ((args.depth * args.aspect_ratio + args.head_dim - 1) // args.head_dim) * args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=args.vocab_size,
        n_layer=args.depth, n_head=model_dim // args.head_dim, n_kv_head=model_dim // args.head_dim,
        n_embd=model_dim, window_pattern=args.window_pattern,
        n_expert=args.n_expert, top_k=args.top_k,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model, config


def main():
    p = argparse.ArgumentParser(description="Benchmark one training step")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--window-pattern", type=str, default="L")
    p.add_argument("--n-expert", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4, help="micro-steps per optimizer step")
    p.add_argument("--warmup", type=int, default=6, help="untimed steps (must cover torch.compile)")
    p.add_argument("--steps", type=int, default=20, help="timed steps")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--budget-hours", type=float, nargs="*", default=[5.0, 6.0, 7.0])
    p.add_argument("--target-tokens", type=float, default=1e9, help="token budget we need to hit")
    args = p.parse_args()

    device_type = autodetect_device_type()
    _, _, _, world_size, device = compute_init(device_type)
    model, config = build(args, device)
    orig_model = model

    gpu_name = torch.cuda.get_device_name(0) if device_type == "cuda" else device_type
    peak_flops = get_peak_flops(gpu_name) if device_type == "cuda" else float("inf")

    total_matmul = orig_model.num_matmul_params()
    active_matmul = orig_model.num_active_matmul_params()
    all_params = orig_model.num_scaling_params()["total"]
    flops_per_token = orig_model.estimate_flops()

    print0("=" * 78)
    print0(f"GPU            : {gpu_name}   peak bf16 {peak_flops:.3e} FLOP/s")
    print0(f"dtype          : {COMPUTE_DTYPE}")
    print0(f"model          : depth {config.n_layer}  n_embd {config.n_embd}  n_head {config.n_head}  "
           f"vocab {config.vocab_size}  seq {config.sequence_len}  window {config.window_pattern}")
    if orig_model.is_moe():
        print0(f"MoE            : {config.n_expert} experts, top-{config.top_k}, Quantile Balancing")
    else:
        print0("MoE            : disabled (dense MLP)")
    print0(f"params         : {all_params:,} total")
    print0(f"matmul params  : {total_matmul:,} total / {active_matmul:,} active "
           f"({active_matmul / total_matmul:.1%})")
    print0(f"FLOPs/token    : {flops_per_token:.4e}  (fwd+bwd, active params only)")

    optimizer = orig_model.setup_optimizer()
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    if not args.no_compile:
        model = torch.compile(model, dynamic=False)

    B, T = args.device_batch_size, args.max_seq_len
    tokens_per_step = B * T * args.grad_accum * world_size
    print0(f"batch          : device {B} x seq {T} x accum {args.grad_accum} = "
           f"{tokens_per_step:,} tokens/step")
    print0("=" * 78)

    # Fixed random batches. Shapes are what matter for timing; content is irrelevant, and
    # reusing a small pool keeps the host out of the measurement.
    # .contiguous() matters: the real dataloader yields contiguous inputs/targets, and
    # GPT.forward does targets.view(-1), which rejects a strided slice.
    gen = torch.Generator(device=device).manual_seed(0)
    pool = []
    for _ in range(4):
        batch = torch.randint(0, config.vocab_size, (B, T + 1), device=device, generator=gen)
        pool.append((batch[:, :-1].contiguous(), batch[:, 1:].contiguous()))

    def one_step(i):
        for micro in range(args.grad_accum):
            x, y = pool[(i + micro) % len(pool)]
            loss = model(x, y) / args.grad_accum
            loss.backward()
        optimizer.step()
        orig_model.apply_qb_update()
        model.zero_grad(set_to_none=True)

    sync = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)

    print0(f"warmup ({args.warmup} steps, includes torch.compile) ...")
    t_warm = time.time()
    for i in range(args.warmup):
        one_step(i)
    sync()
    print0(f"warmup done in {time.time() - t_warm:.1f}s")

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print0(f"timing {args.steps} steps ...")
    times = []
    for i in range(args.steps):
        sync()
        t0 = time.time()
        one_step(args.warmup + i)
        sync()
        times.append(time.time() - t0)

    times.sort()
    median = times[len(times) // 2]
    fastest, slowest = times[0], times[-1]
    tok_per_sec = tokens_per_step / median
    achieved_flops = flops_per_token * tok_per_sec
    mfu = achieved_flops / peak_flops

    print0("=" * 78)
    print0(f"step time      : {median * 1000:.1f} ms median   ({fastest * 1000:.1f} min / {slowest * 1000:.1f} max)")
    print0(f"throughput     : {tok_per_sec:,.0f} tokens/sec")
    print0(f"achieved       : {achieved_flops:.3e} FLOP/s   MFU {mfu:.1%}")
    if device_type == "cuda":
        print0(f"peak memory    : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB of "
               f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    print0("-" * 78)
    for hours in args.budget_hours:
        tokens = tok_per_sec * hours * 3600
        verdict = "OK " if tokens >= args.target_tokens else "SHORT"
        print0(f"  {hours:g}h  ->  {tokens / 1e9:6.2f}B tokens   [{verdict} vs {args.target_tokens / 1e9:.2f}B target]")
    need_hours = args.target_tokens / tok_per_sec / 3600
    print0(f"  {args.target_tokens / 1e9:.2f}B tokens needs {need_hours:.2f}h of pure step time")
    print0("=" * 78)
    print0("NB: random tokens, no dataloader. The real run also pays CPU-side BPE tokenization,")
    print0("    so treat this as the ceiling and confirm with scripts.base_train.")


if __name__ == "__main__":
    main()
