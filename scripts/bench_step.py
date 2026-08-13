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
        n_expert=args.n_expert, top_k=args.top_k, moe_dispatch=args.moe_dispatch,
        moe_capacity_factor=args.moe_capacity_factor,
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
    p.add_argument("--moe-dispatch", type=str, default="loop", choices=["loop", "batched", "compiled", "permute", "grouped"],
                   help="expert execution: per-expert python loop, or one batched bmm over all experts")
    p.add_argument("--moe-capacity-factor", type=float, default=1.25,
                   help="only for --moe-dispatch=compiled: per-expert slots as a multiple of the balanced share")
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4, help="micro-steps per optimizer step")
    p.add_argument("--warmup", type=int, default=6, help="untimed steps (must cover torch.compile)")
    p.add_argument("--steps", type=int, default=20, help="timed steps")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="run torch.profiler over a few steps and print where the GPU time goes")
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
    # Capacity-limited dispatch silently discards tokens when an expert overflows. That is a
    # quality cost hiding inside a speed number, so report it rather than assume QB prevents it.
    if orig_model.is_moe() and args.moe_dispatch == "compiled":
        from nanochat.gpt import MoEMLP
        moes = [b.mlp for b in orig_model.transformer.h if isinstance(b.mlp, MoEMLP)]
        dropped = sum(m.dropped_tokens.item() for m in moes)
        routed = args.device_batch_size * args.max_seq_len * args.top_k * len(moes) * \
                 args.grad_accum * (args.warmup + args.steps)
        print0(f"dropped tokens : {dropped:,} of {routed:,} routed = {dropped / max(routed,1):.3%} "
               f"(capacity factor {args.moe_capacity_factor})")
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

    if args.profile:
        profile_steps(model, orig_model, optimizer, pool, args, device_type)


def profile_steps(model, orig_model, optimizer, pool, args, device_type):
    """Where does the step actually go? MFU says 'not into matmuls' but not what instead.

    Groups CUDA kernels by what they are rather than by name, because the raw top-20 is a
    wall of `void at::native::vectorized_elementwise_kernel<...>` that tells you nothing.
    """
    from torch.profiler import ProfilerActivity, profile

    def one(i):
        for micro in range(args.grad_accum):
            x, y = pool[(i + micro) % len(pool)]
            (model(x, y) / args.grad_accum).backward()
        optimizer.step()
        orig_model.apply_qb_update()
        model.zero_grad(set_to_none=True)

    n_prof = 3
    print0(f"\nprofiling {n_prof} steps ...")
    if device_type == "cuda":
        torch.cuda.synchronize()
    t_prof = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for i in range(n_prof):
            one(1000 + i)
        if device_type == "cuda":
            torch.cuda.synchronize()
    wall_s = time.time() - t_prof

    # --- Is the GPU actually idle waiting on the host? ---
    # Sum ONLY true device kernels. key_averages() also carries aten::/Optimizer entries whose
    # self_device_time_total re-counts their children, which is why an earlier version of this
    # reported more CUDA time than wall clock. Real kernels are the ones the profiler tags with
    # DeviceType.CUDA and whose names are the launched symbols, not operator names.
    from torch.autograd import DeviceType
    kernels = [e for e in prof.key_averages()
               if e.device_type == DeviceType.CUDA and not e.key.startswith(("aten::", "Optimizer", "## ", "cud"))]
    busy_us = sum(e.self_device_time_total for e in kernels)
    wall_us = wall_s * 1e6
    print0("=" * 78)
    print0("GPU OCCUPANCY  (is the host starving the device?)")
    print0(f"  wall clock over {n_prof} steps : {wall_s:8.3f}s")
    print0(f"  summed device-kernel time   : {busy_us / 1e6:8.3f}s")
    print0(f"  GPU busy                    : {busy_us / wall_us:8.1%}")
    print0(f"  GPU idle (host-bound)       : {1 - busy_us / wall_us:8.1%}")
    print0(f"  distinct kernels launched   : {sum(e.count for e in kernels):,} "
           f"({sum(e.count for e in kernels) / n_prof:,.0f} per step)")
    print0("=" * 78 + "\n")

    evts = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in evts)
    print0(f"attribution below double-counts operator vs kernel entries; read shares, not totals")
    print0(f"total (double-counted) over {n_prof} steps: {total / 1e6:.2f}s\n")

    # bucket by kernel family. order matters: first match wins.
    BUCKETS = [
        ("matmul / GEMM",      ("gemm", "cutlass", "s16816", "wgrad", "dgrad", "bmm", "ampere", "sm90", "cublas")),
        ("attention (SDPA)",   ("flash", "attention", "sdpa", "efficient_attention", "mha")),
        ("elementwise / cast", ("elementwise", "vectorized", "copy", "cast", "convert", "fill", "add", "mul")),
        ("reduction / softmax",("softmax", "reduce", "norm", "sum", "mean", "logsumexp", "cross_entropy", "nll")),
        ("index / gather",     ("index", "gather", "scatter", "sort", "bincount", "cumsum", "nonzero", "where", "argsort")),
        ("optimizer",          ("adam", "muon", "foreach", "lerp", "newton", "polar", "clamp")),
    ]
    tally, examples = {}, {}
    for e in evts:
        name = e.key.lower()
        label = next((b for b, keys in BUCKETS if any(k in name for k in keys)), "other")
        tally[label] = tally.get(label, 0) + e.self_device_time_total
        if e.self_device_time_total > examples.get(label, (0, ""))[0]:
            examples[label] = (e.self_device_time_total, e.key[:58])
    print0(f"{'bucket':22s} {'CUDA time':>10s} {'share':>7s}  largest single kernel")
    for label, t in sorted(tally.items(), key=lambda kv: -kv[1]):
        print0(f"{label:22s} {t / 1e6:9.2f}s {t / total:6.1%}  {examples[label][1]}")

    print0("\ntop 12 individual kernels by self CUDA time:")
    for e in sorted(evts, key=lambda e: -e.self_device_time_total)[:12]:
        print0(f"  {e.self_device_time_total / 1e6:7.3f}s {e.self_device_time_total / total:6.1%}  {e.key[:70]}")


if __name__ == "__main__":
    main()
