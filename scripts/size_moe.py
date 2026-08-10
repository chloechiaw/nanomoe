"""
Size a nanoMoE config: total vs. active params, FLOPs/token, and what that buys you in a
5-7 hour single-A10G budget.

Pure python (no torch), so you can iterate on it locally in a second. The accounting mirrors
nanochat/gpt.py exactly; pass --verify on a machine with torch to check it against a real
meta-device model build.

Examples:

    # the brief's target: ~40M total / ~15M active, 8 experts top-2
    python -m scripts.size_moe --d-model 320 --n-layer 8 --n-expert 8 --top-k 2

    # same trunk, but with the smaller vocab that keeps embeddings from dominating
    python -m scripts.size_moe --d-model 320 --n-layer 8 --vocab-size 8192

    # sweep d_model to find what hits the target
    python -m scripts.size_moe --sweep
"""

import argparse

# Dense bf16 peak for the GA102 part Modal hands out. Note that `gpu="A10G"` has been observed
# to resolve to a plain A10 (72 SM vs 80), so this uses the A10 number as the conservative case.
# See the comment in nanochat/common.py:get_peak_flops for where 125e12 comes from.
A10_PEAK_FLOPS = 125e12


def size(
    d_model,
    n_layer,
    vocab_size=32768,
    head_dim=64,
    seq_len=2048,
    ffn_mult=4,
    n_expert=1,
    top_k=1,
    window_pattern="L",
):
    """Return a dict of parameter counts and FLOPs/token for a nanochat/nanoMoE config.

    n_expert=1, top_k=1 reproduces the dense nanochat model exactly.
    """
    assert d_model % head_dim == 0, f"d_model {d_model} must be divisible by head_dim {head_dim}"
    assert top_k <= n_expert
    n_head = d_model // head_dim
    n_kv_head = n_head  # nanochat trains with full MHA (n_kv_head == n_head)
    kv_dim = n_kv_head * head_dim
    padded_vocab = ((vocab_size + 63) // 64) * 64
    d_ff = ffn_mult * d_model

    # Value embeddings live on alternating layers, last layer always included (gpt.py has_ve)
    ve_layers = [i for i in range(n_layer) if i % 2 == (n_layer - 1) % 2]

    # --- per-layer matmul params ---
    attn = 2 * d_model * d_model + 2 * d_model * kv_dim  # c_q, c_proj, c_k, c_v
    expert = 2 * d_model * d_ff  # c_fc + c_proj for ONE expert
    router = d_model * n_expert if n_expert > 1 else 0

    ffn_total = n_expert * expert + router
    ffn_active = top_k * expert + router

    per_layer_total = attn + ffn_total
    per_layer_active = attn + ffn_active

    ve_gate = 12 * n_kv_head * len(ve_layers)
    smear_gate = 24

    trunk_total = n_layer * per_layer_total + ve_gate
    trunk_active = n_layer * per_layer_active + ve_gate

    lm_head = padded_vocab * d_model
    wte = padded_vocab * d_model
    value_embeds = len(ve_layers) * padded_vocab * kv_dim
    scalars = 2 * n_layer + 1 + 1  # resid_lambdas, x0_lambdas, smear_lambda, backout_lambda

    embeddings = wte + value_embeds

    # --- attention FLOPs, honouring the sliding-window pattern ---
    short_window = -(-seq_len // 4 // 128) * 128
    windows = []
    for i in range(n_layer):
        windows.append(seq_len if window_pattern.upper()[i % len(window_pattern)] == "L" else short_window)
    windows[-1] = seq_len  # final layer is always full context
    attn_flops = sum(12 * n_head * head_dim * min(w, seq_len) for w in windows)

    # matmul params that see the token stream: trunk + lm_head + smear_gate
    matmul_total = trunk_total + lm_head + smear_gate
    matmul_active = trunk_active + lm_head + smear_gate

    return {
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": n_head,
        "d_ff": d_ff,
        "n_expert": n_expert,
        "top_k": top_k,
        "vocab_size": vocab_size,
        "padded_vocab": padded_vocab,
        # params
        "wte": wte,
        "value_embeds": value_embeds,
        "lm_head": lm_head,
        "trunk_total": trunk_total,
        "trunk_active": trunk_active,
        "scalars": scalars,
        "params_total": trunk_total + lm_head + embeddings + smear_gate + scalars,
        "params_active": trunk_active + lm_head + embeddings + smear_gate + scalars,
        "nonembed_total": trunk_total + lm_head + smear_gate + scalars,
        "nonembed_active": trunk_active + lm_head + smear_gate + scalars,
        # flops (6 per matmul param for fwd+bwd, plus attention; see gpt.estimate_flops)
        "flops_per_token": 6 * matmul_active + attn_flops,
        "flops_per_token_dense_equiv": 6 * matmul_total + attn_flops,
    }


def report(s, mfu=0.30, hours=6.0, peak=A10_PEAK_FLOPS):
    m = lambda x: f"{x / 1e6:9.2f}M"
    active_frac = s["nonembed_active"] / s["nonembed_total"]
    tok_per_sec = mfu * peak / s["flops_per_token"]
    tokens = tok_per_sec * hours * 3600

    print(
        f"d_model={s['d_model']}  n_layer={s['n_layer']}  n_head={s['n_head']}  "
        f"d_ff={s['d_ff']}  experts={s['n_expert']} top-{s['top_k']}  vocab={s['vocab_size']}"
    )
    print("  params")
    print(f"    wte                  {m(s['wte'])}")
    print(f"    value_embeds         {m(s['value_embeds'])}")
    print(f"    lm_head              {m(s['lm_head'])}")
    print(f"    trunk (total)        {m(s['trunk_total'])}")
    print(f"    trunk (active)       {m(s['trunk_active'])}")
    print(f"    TOTAL                {m(s['params_total'])}     (incl. embeddings)")
    print(f"    ACTIVE               {m(s['params_active'])}     (incl. embeddings)")
    print(f"    total  ex-embeddings {m(s['nonembed_total'])}")
    print(f"    active ex-embeddings {m(s['nonembed_active'])}   -> active frac {active_frac:.1%}")
    print("  compute")
    print(f"    FLOPs/token          {s['flops_per_token']:.3e}")
    print(f"    A10 @ {mfu:.0%} MFU       {tok_per_sec:,.0f} tok/s  ->  {tokens / 1e6:,.0f}M tokens in {hours:g}h")
    print()


def verify(s):
    """Cross-check the dense (n_expert=1) numbers against a real meta-device GPT build."""
    import torch

    from nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig(
        sequence_len=2048,
        vocab_size=s["vocab_size"],
        n_layer=s["n_layer"],
        n_head=s["n_head"],
        n_kv_head=s["n_head"],
        n_embd=s["d_model"],
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(cfg)
    real = model.num_scaling_params()
    print("verify against nanochat/gpt.py (dense):")
    print(f"    wte           model={real['wte']:,}  calc={s['wte']:,}")
    print(f"    value_embeds  model={real['value_embeds']:,}  calc={s['value_embeds']:,}")
    print(f"    lm_head       model={real['lm_head']:,}  calc={s['lm_head']:,}")
    print(f"    trunk         model={real['transformer_matrices']:,}  calc={s['trunk_total']:,}")
    print(f"    total         model={real['total']:,}  calc={s['params_total']:,}")
    print(f"    flops/token   model={model.estimate_flops():.3e}  calc={s['flops_per_token']:.3e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=320)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--ffn-mult", type=int, default=4, help="d_ff = ffn_mult * d_model, per expert")
    p.add_argument("--n-expert", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--window-pattern", type=str, default="L")
    p.add_argument("--mfu", type=float, default=0.30, help="assumed MFU; measure it with `modal run modal_app.py::smoke`")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--sweep", action="store_true", help="sweep d_model x n_layer against the brief's target")
    p.add_argument("--verify", action="store_true", help="check the dense accounting against a real model (needs torch)")
    a = p.parse_args()

    common = dict(
        vocab_size=a.vocab_size,
        head_dim=a.head_dim,
        seq_len=a.seq_len,
        ffn_mult=a.ffn_mult,
        window_pattern=a.window_pattern,
    )

    if a.sweep:
        print(f"target: ~30-60M total / ~10-20M active, ex-embeddings (vocab={a.vocab_size})\n")
        for n_layer in (6, 8, 10, 12):
            for d_model in (192, 256, 320, 384, 448):
                s = size(d_model, n_layer, n_expert=a.n_expert, top_k=a.top_k, **common)
                hit = 30e6 <= s["nonembed_total"] <= 60e6 and 10e6 <= s["nonembed_active"] <= 20e6
                tok = a.mfu * A10_PEAK_FLOPS / s["flops_per_token"] * a.hours * 3600
                print(
                    f"  {'*' if hit else ' '} d={d_model:4d} L={n_layer:2d}  "
                    f"total {s['nonembed_total'] / 1e6:6.1f}M  active {s['nonembed_active'] / 1e6:6.1f}M  "
                    f"({s['nonembed_active'] / s['nonembed_total']:.0%})  "
                    f"embed {(s['wte'] + s['value_embeds'] + s['lm_head']) / 1e6:6.1f}M  "
                    f"{tok / 1e6:6.0f}M tok/{a.hours:g}h"
                )
            print()
        print("  * = inside the brief's target band (ex-embeddings)")
        return

    s = size(a.d_model, a.n_layer, n_expert=a.n_expert, top_k=a.top_k, **common)
    report(s, mfu=a.mfu, hours=a.hours)

    if a.n_expert > 1:
        d = size(a.d_model, a.n_layer, n_expert=1, top_k=1, **common)
        print("same trunk, dense (for reference):")
        report(d, mfu=a.mfu, hours=a.hours)

    if a.verify:
        verify(size(a.d_model, a.n_layer, n_expert=1, top_k=1, **common))


if __name__ == "__main__":
    main()
