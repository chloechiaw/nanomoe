"""
Tests for the MoE block and Quantile Balancing.

Run on CPU: python -m pytest tests/test_moe.py -v
"""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig, MLP, MoEMLP


def make_config(n_expert=8, top_k=2, n_layer=2, n_embd=64, vocab_size=256):
    return GPTConfig(
        sequence_len=64, vocab_size=vocab_size, n_layer=n_layer,
        n_head=2, n_kv_head=2, n_embd=n_embd, window_pattern="L",
        n_expert=n_expert, top_k=top_k,
    )


def build(config):
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def maxvio(counts):
    """(max_load - mean_load) / mean_load, per row. 0 == perfectly balanced."""
    counts = counts.float()
    mean = counts.mean(dim=-1)
    return ((counts.max(dim=-1).values - mean) / mean)


# -----------------------------------------------------------------------------
# structure


def test_dense_when_single_expert():
    model = build(make_config(n_expert=1))
    assert all(isinstance(b.mlp, MLP) for b in model.transformer.h)
    assert not model.is_moe()


def test_moe_replaces_mlp():
    model = build(make_config())
    assert all(isinstance(b.mlp, MoEMLP) for b in model.transformer.h)
    assert model.is_moe()


def test_top_k_must_leave_room_for_the_qb_threshold():
    # QB needs a (k+1)-th logit, so top_k == n_expert is not representable
    with pytest.raises(AssertionError):
        build(make_config(n_expert=4, top_k=4))


def test_active_param_accounting():
    n_expert, top_k = 8, 2
    model = build(make_config(n_expert=n_expert, top_k=top_k))
    total = model.num_matmul_params()
    active = model.num_active_matmul_params()
    # one expert's worth of params, times the number of layers, times the inactive count
    from nanochat.gpt import Linear
    per_expert = sum(m.weight.numel() for m in model.transformer.h[0].mlp.experts[0].modules()
                     if isinstance(m, Linear))
    expected = total - (n_expert - top_k) * per_expert * len(model.transformer.h)
    assert active == expected
    assert active < total

    # a dense model must be unaffected by the new code path
    dense = build(make_config(n_expert=1))
    assert dense.num_active_matmul_params() == dense.num_matmul_params()


def test_scaling_param_bookkeeping_still_balances():
    # GPT.num_scaling_params() asserts its own groups sum to the true parameter count;
    # the router and expert weights must land inside transformer_matrices.
    model = build(make_config())
    counts = model.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------------
# forward / backward


def test_forward_shape_and_finiteness():
    config = make_config()
    model = build(config)
    idx = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    logits = model(idx)
    assert logits.shape == (2, config.sequence_len, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_router_gradient_is_zero_at_init():
    """Documents a real interaction with nanochat's init rather than a defect: every c_proj
    starts at zero, so expert outputs are zero, so dL/dgate is zero and the router has no
    gradient on step 0. QB is unaffected (it reads logits, not gradients)."""
    config = make_config()
    model = build(config)
    idx = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    model(idx, targets=idx).backward()
    for block in model.transformer.h:
        assert block.mlp.router.weight.grad.abs().sum().item() == 0.0


def test_router_receives_gradient_once_experts_are_live():
    """Selection is non-differentiable, but the sigmoid combine weights are the router's
    gradient path. If this breaks, the router silently never learns for the whole run."""
    torch.manual_seed(0)
    config = make_config()
    model = build(config)
    # stand in for "one optimizer step has happened": nudge c_proj off its zero init
    with torch.no_grad():
        for block in model.transformer.h:
            for expert in block.mlp.experts:
                expert.c_proj.weight.normal_(std=0.02)

    idx = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    model(idx, targets=idx).backward()
    for block in model.transformer.h:
        g = block.mlp.router.weight.grad
        assert g is not None, "router has no grad at all"
        assert torch.isfinite(g).all().item(), "router grad is not finite"
        assert g.abs().sum().item() > 0, "router grad is identically zero"


def test_expert_with_zero_tokens_still_gets_a_gradient():
    """Muon does torch.stack([p.grad for p in group]); a None grad from an unrouted expert
    would crash the optimizer step. An empty matmul must still produce a zero grad."""
    config = make_config(n_expert=8, top_k=2)
    model = build(config)
    # Force every token to experts 0 and 1 by making their bias overwhelming
    for block in model.transformer.h:
        block.mlp.router_bias.copy_(torch.tensor([1e4, 1e4, 0.0, 0, 0, 0, 0, 0]))
    idx = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    model(idx, targets=idx).backward()

    for block in model.transformer.h:
        counts = block.mlp.expert_counts
        assert counts[2:].sum() == 0, "test setup failed: experts 2+ should be starved"
        for e, expert in enumerate(block.mlp.experts):
            for name, p in expert.named_parameters():
                assert p.grad is not None, f"expert {e} {name} has no grad"
        # starved experts must get exactly zero, not garbage
        assert block.mlp.experts[3].c_fc.weight.grad.abs().sum() == 0


@pytest.mark.parametrize("mode", ["batched", "permute", "grouped"])
def test_all_dispatch_modes_match_loop(mode):
    """Every *dropless* dispatch variant must be arithmetically identical to the reference
    loop. They differ only in memory-access pattern, so any drift means a speed number is
    measuring the wrong computation.

    "compiled" is deliberately excluded: it pads to a fixed capacity, so tokens overflowing a
    full expert are dropped and its output legitimately differs from the loop whenever routing
    is imbalanced (e.g. at init, before QB has converged). That is the tradeoff it makes for
    static shapes -- and having measured it ~2x slower, the tradeoff buys nothing."""
    torch.manual_seed(0)
    ref_cfg = make_config(n_expert=8, top_k=2, n_layer=2, n_embd=64)
    alt_cfg = make_config(n_expert=8, top_k=2, n_layer=2, n_embd=64)
    alt_cfg.moe_dispatch = mode

    torch.manual_seed(1234)
    ref = build(ref_cfg)
    torch.manual_seed(1234)
    alt = build(alt_cfg)
    with torch.no_grad():
        for blk in ref.transformer.h:
            for e in blk.mlp.experts:
                e.c_proj.weight.normal_(std=0.02)   # off the zero init so experts contribute
    alt.load_state_dict(ref.state_dict())
    ref.eval(); alt.eval()

    idx = torch.randint(0, ref_cfg.vocab_size, (3, ref_cfg.sequence_len))
    with torch.no_grad():
        torch.testing.assert_close(ref(idx), alt(idx), rtol=1e-4, atol=1e-5)


def test_batched_dispatch_matches_loop():
    """The kernel's whole justification is that it changes only *how many* kernels launch,
    not the math. If this drifts, every speed number it produces is meaningless."""
    torch.manual_seed(0)
    cfg_loop = make_config(n_expert=8, top_k=2, n_layer=2, n_embd=64)
    cfg_batched = make_config(n_expert=8, top_k=2, n_layer=2, n_embd=64)
    cfg_batched.moe_dispatch = "batched"

    torch.manual_seed(1234)
    a = build(cfg_loop)
    torch.manual_seed(1234)
    b = build(cfg_batched)
    b.load_state_dict(a.state_dict())          # identical weights, different dispatch
    # nudge c_proj off its zero init so the experts actually contribute
    with torch.no_grad():
        for m in (a, b):
            for blk in m.transformer.h:
                for e in blk.mlp.experts:
                    e.c_proj.weight.normal_(std=0.02)
        b.load_state_dict(a.state_dict())
    a.eval(); b.eval()

    idx = torch.randint(0, cfg_loop.vocab_size, (3, cfg_loop.sequence_len))
    with torch.no_grad():
        torch.testing.assert_close(a(idx), b(idx), rtol=1e-4, atol=1e-5)


def test_batched_dispatch_is_dropless_under_imbalance():
    """Padding is to max(counts), not a fixed capacity, so even a badly skewed router must
    still route every token. A capacity-factor implementation would silently drop here."""
    torch.manual_seed(0)
    cfg = make_config(n_expert=8, top_k=2, n_layer=1, n_embd=64)
    cfg.moe_dispatch = "batched"
    model = build(cfg)
    moe = model.transformer.h[0].mlp
    with torch.no_grad():
        for e in moe.experts:
            e.c_proj.weight.normal_(std=0.02)
        moe.router_bias.copy_(torch.tensor([6.0, 3.0, 0, 0, 0, 0, 0, -6.0]))  # hard skew

    idx = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
    model(idx)
    counts = moe.expert_counts
    n_tok = 2 * cfg.sequence_len * cfg.top_k
    assert counts.sum().item() == n_tok, f"dropped tokens: {counts.sum().item()} != {n_tok}"


def test_dispatch_matches_a_reference_dense_computation():
    """The gather/scatter loop must equal the obvious (slow) masked formulation."""
    torch.manual_seed(0)
    config = make_config(n_expert=6, top_k=2, n_embd=32)
    moe = MoEMLP(config)
    for expert in moe.experts:
        torch.nn.init.normal_(expert.c_fc.weight, std=0.1)
        torch.nn.init.normal_(expert.c_proj.weight, std=0.1)
    torch.nn.init.normal_(moe.router.weight, std=0.5)
    moe.router_bias.normal_(std=0.5)
    moe.eval()

    x = torch.randn(2, 16, config.n_embd)
    got = moe(x)

    # reference: run every token through every expert, mask, weight, sum
    xf = x.view(-1, config.n_embd)
    logits = moe.router(xf).float()
    biased = logits + moe.router_bias
    _, topk_idx = torch.topk(biased, config.top_k + 1, dim=-1)
    idx = topk_idx[:, :config.top_k]
    gate = torch.sigmoid(logits.gather(1, idx)).to(x.dtype)
    want = torch.zeros_like(xf)
    for e, expert in enumerate(moe.experts):
        y = expert(xf)
        for slot in range(config.top_k):
            sel = (idx[:, slot] == e)
            want[sel] += y[sel] * gate[sel, slot].unsqueeze(-1)

    torch.testing.assert_close(got, want.view_as(x), rtol=1e-4, atol=1e-5)


# -----------------------------------------------------------------------------
# Quantile Balancing


def test_qb_bias_is_mean_centred_and_not_a_parameter():
    model = build(make_config())
    moe = model.transformer.h[0].mlp
    assert not isinstance(moe.router_bias, torch.nn.Parameter)
    assert "router_bias" in model.state_dict().keys().__str__()  # persistent: survives a resume

    idx = torch.randint(0, 256, (2, 64))
    model(idx, targets=idx)
    model.apply_qb_update()
    assert abs(moe.router_bias.mean().item()) < 1e-5
    assert moe.router_bias.abs().sum() > 0, "bias should have moved off zero"


def test_qb_reduces_load_imbalance():
    """The load-balancing claim itself. Start from a deliberately skewed router and check
    that repeated QB updates drive MaxVio down without any tunable knob."""
    torch.manual_seed(0)
    config = make_config(n_expert=8, top_k=2, n_layer=1, n_embd=64)
    model = build(config)
    moe = model.transformer.h[0].mlp
    # Skew via the routing bias itself, which is the exact channel QB controls: a large positive
    # bias makes expert 0 win essentially every token, the worst case QB has to recover from.
    # (Perturbing the router *weights* is a poor way to force this — adding a constant to a row
    # washes out against the rms-normed, roughly zero-mean residual stream, and scaling a row
    # only inflates that logit's variance, making it extreme in both directions and so selected
    # about half the time.)
    with torch.no_grad():
        moe.router_bias.copy_(torch.tensor([8.0, 0, 0, 0, 0, 0, 0, 0]))

    idx = torch.randint(0, config.vocab_size, (4, config.sequence_len))

    model(idx, targets=idx)
    before = maxvio(model.expert_load(reset=True))[0].item()

    for _ in range(30):
        model(idx, targets=idx)
        model.apply_qb_update()
        model.expert_load(reset=True)  # discard, we only want the final steady state

    model(idx, targets=idx)
    after = maxvio(model.expert_load(reset=True))[0].item()

    # MaxVio ranges over [0, n_expert/top_k - 1] = [0, 3] for this config
    assert before > 1.5, f"test setup failed, router was not skewed enough (MaxVio {before:.3f})"
    assert after < before / 4, f"QB failed to balance: MaxVio {before:.3f} -> {after:.3f}"
    assert after < 0.3, f"QB left too much imbalance: MaxVio {after:.3f}"


def test_qb_accumulates_across_micro_steps():
    """Under gradient accumulation the betas must average over micro-batches, which is the
    single-GPU stand-in for the reference's pmean across data-parallel shards."""
    config = make_config(n_expert=8, top_k=2, n_layer=1)
    model = build(config)
    moe = model.transformer.h[0].mlp

    for _ in range(3):
        model(torch.randint(0, config.vocab_size, (2, config.sequence_len)))
    assert moe.qb_beta_count.item() == 3
    model.apply_qb_update()
    assert moe.qb_beta_count.item() == 0


def test_no_qb_accumulation_in_eval():
    config = make_config(n_layer=1)
    model = build(config)
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, config.vocab_size, (2, config.sequence_len)))
    assert model.transformer.h[0].mlp.qb_beta_count.item() == 0
