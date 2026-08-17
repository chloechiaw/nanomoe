"""
Hand-written Triton kernels for the MoE hot path.

Stage A: the combine. After the expert GEMMs produce y (one row per routed (token, expert)
pair, in expert-major order), the eager tail was three kernels:

    y = y * gate_o.unsqueeze(-1)      # read y, read gate, write y'
    out = torch.zeros_like(xf)        # write zeros
    out.index_add_(0, rows, y)        # read y', read+write out

This file does it in one kernel each way. The forward iterates tokens rather than pairs:
token t's k contributions sit at sorted positions inv[t*k + s], so each output row is a
plain sum of k loads. No atomics, so runs are deterministic.

Triton only exists on CUDA, and the tests run on CPU, so every entry point falls back to
the eager formulation off-GPU. The eager path is also the correctness reference.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _combine_fwd(y_ptr, gate_ptr, inv_ptr, out_ptr, C, K: tl.constexpr, BLOCK_C: tl.constexpr):
        t = tl.program_id(0)
        cb = tl.program_id(1)
        cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = cols < C
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for s in tl.static_range(K):
            p = tl.load(inv_ptr + t * K + s)
            g = tl.load(gate_ptr + p).to(tl.float32)
            yv = tl.load(y_ptr + p * C + cols, mask=cmask, other=0.0).to(tl.float32)
            acc += g * yv
        tl.store(out_ptr + t * C + cols, acc.to(out_ptr.dtype.element_ty), mask=cmask)

    @triton.jit
    def _combine_bwd(dout_ptr, y_ptr, gate_ptr, rows_ptr, dy_ptr, dgate_ptr, C, BLOCK_C: tl.constexpr):
        # One program per routed pair p: dy[p] = gate[p] * dout[rows[p]], and
        # dgate[p] = <y[p], dout[rows[p]]>, both in a single pass over C.
        p = tl.program_id(0)
        r = tl.load(rows_ptr + p)
        g = tl.load(gate_ptr + p).to(tl.float32)
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for cb in range(0, tl.cdiv(C, BLOCK_C)):
            cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
            cmask = cols < C
            do = tl.load(dout_ptr + r * C + cols, mask=cmask, other=0.0).to(tl.float32)
            yv = tl.load(y_ptr + p * C + cols, mask=cmask, other=0.0).to(tl.float32)
            tl.store(dy_ptr + p * C + cols, (g * do).to(dy_ptr.dtype.element_ty), mask=cmask)
            acc += do * yv
        tl.store(dgate_ptr + p, tl.sum(acc, axis=0))

    class _GateCombine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y, gate_o, inv, rows, n_tokens):
            C = y.shape[1]
            K = inv.numel() // n_tokens
            out = torch.empty(n_tokens, C, device=y.device, dtype=y.dtype)
            _combine_fwd[(n_tokens, triton.cdiv(C, 256))](y, gate_o, inv, out, C, K=K, BLOCK_C=256)
            ctx.save_for_backward(y, gate_o, rows)
            return out

        @staticmethod
        def backward(ctx, dout):
            y, gate_o, rows = ctx.saved_tensors
            P, C = y.shape
            dy = torch.empty_like(y)
            dgate = torch.empty(P, device=y.device, dtype=torch.float32)
            _combine_bwd[(P,)](dout.contiguous(), y, gate_o, rows, dy, dgate, C, BLOCK_C=256)
            return dy, dgate.to(gate_o.dtype), None, None, None


def gate_combine(y, gate_o, order, rows, n_tokens):
    """out[t] = sum over the k pairs routed from token t of gate * expert output."""
    if HAS_TRITON and y.is_cuda:
        # inv[p_orig] = sorted position of original pair p_orig, so token t's contributions
        # are at inv[t*k + s]. Deterministic sum instead of index_add_'s atomics.
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return _GateCombine.apply(y, gate_o, inv.to(torch.int32), rows.to(torch.int32), n_tokens)
    out = torch.zeros(n_tokens, y.shape[1], device=y.device, dtype=y.dtype)
    out.index_add_(0, rows, y * gate_o.unsqueeze(-1))
    return out
