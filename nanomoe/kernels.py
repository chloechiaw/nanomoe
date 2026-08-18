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

    @torch.library.triton_op("nanomoe::gate_combine", mutates_args={})
    def _gate_combine_op(y: torch.Tensor, gate_o: torch.Tensor, inv: torch.Tensor,
                         rows: torch.Tensor, n_tokens: int) -> torch.Tensor:
        C = y.shape[1]
        K = inv.numel() // n_tokens
        out = torch.empty(n_tokens, C, device=y.device, dtype=y.dtype)
        torch.library.wrap_triton(_combine_fwd)[(n_tokens, triton.cdiv(C, 256))](y, gate_o, inv, out, C, K=K, BLOCK_C=256)
        return out

    @_gate_combine_op.register_fake
    def _(y, gate_o, inv, rows, n_tokens):
        return y.new_empty(n_tokens, y.shape[1])

    def _gate_combine_setup(ctx, inputs, output):
        y, gate_o, inv, rows, n_tokens = inputs
        ctx.save_for_backward(y, gate_o, rows)

    def _gate_combine_bwd(ctx, dout):
        y, gate_o, rows = ctx.saved_tensors
        P, C = y.shape
        dy = torch.empty_like(y)
        dgate = torch.empty(P, device=y.device, dtype=torch.float32)
        torch.library.wrap_triton(_combine_bwd)[(P,)](dout.contiguous(), y, gate_o, rows, dy, dgate, C, BLOCK_C=256)
        return dy, dgate.to(gate_o.dtype), None, None, None

    _gate_combine_op.register_autograd(_gate_combine_bwd, setup_context=_gate_combine_setup)


def gate_combine(y, gate_o, order, rows, n_tokens, inv=None):
    """out[t] = sum over the k pairs routed from token t of gate * expert output."""
    if HAS_TRITON and y.is_cuda:
        # inv[p_orig] = sorted position of original pair p_orig, so token t's contributions
        # are at inv[t*k + s]. Deterministic sum instead of index_add_'s atomics.
        if inv is None:
            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=order.device)
            inv = inv.to(torch.int32)
        return torch.ops.nanomoe.gate_combine(y, gate_o, inv, rows.to(torch.int32), n_tokens)
    out = torch.zeros(n_tokens, y.shape[1], device=y.device, dtype=y.dtype)
    out.index_add_(0, rows, y * gate_o.unsqueeze(-1))
    return out


# -----------------------------------------------------------------------------
# Stage B: the up-projection, h = relu(xf[rows] @ W1[e].T)^2, on QuACK's grouped GEMMs.
#
# A hand-written Triton version of this lost to CUTLASS end-to-end (27.3% vs 30.5% MFU): the
# fused forward won but the hand dW kernel gave it all back. Lesson applied here, the same
# one sonic-moe's architecture teaches: never author the GEMM core. QuACK (Dao-AILab) exposes
# everything this needs as arguments: relu_sq as a registered epilogue, cu_seqlens_m for the
# ragged expert segments, A_idx to fuse the gather, and cu_seqlens_k + A_idx for the dW GEMM
# that reduces over each expert's pairs.
#
# Backward never sees the pre-activation: relu(s) = sqrt(h) wherever h > 0, so
# ds = dh * 2*sqrt(h) comes from the output we keep anyway.
#
# Both directions are wrapped as opaque custom ops so neither dynamo nor AOTAutograd ever
# traces QuACK's python (autotuner and all); they only see two graph nodes with fake impls.

try:
    from quack.gemm_interface import gemm as _qk_gemm, gemm_act as _qk_gemm_act
    HAS_QUACK = torch.cuda.is_available()
except Exception:
    HAS_QUACK = False

if HAS_TRITON:

    @triton.jit
    def _drelu_sq(dh_ptr, h_ptr, ds_ptr, numel, BLOCK: tl.constexpr):
        # ds = dh * 2*sqrt(h): relu(s) = sqrt(h) wherever h > 0, so this is the whole
        # relu^2 backward in one pass. Lives here because it runs inside an opaque custom
        # op where Inductor cannot fuse eager elementwise chains.
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < numel
        dh = tl.load(dh_ptr + i, mask=m, other=0.0).to(tl.float32)
        h = tl.load(h_ptr + i, mask=m, other=0.0).to(tl.float32)
        tl.store(ds_ptr + i, (dh * 2.0 * tl.sqrt(h)).to(ds_ptr.dtype.element_ty), mask=m)


if HAS_QUACK:

    @torch.library.custom_op("nanomoe::quack_up", mutates_args=())
    def _quack_up_op(xf: torch.Tensor, w_fc: torch.Tensor, rows: torch.Tensor,
                     offs0: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        wb = w_fc.to(xf.dtype)
        _, h = _qk_gemm_act(
            xf, wb.permute(0, 2, 1),                     # B is (E, K=C, N=DFF)
            activation="relu_sq",
            cu_seqlens_m=offs0, A_idx=rows,
            store_preact=False,
        )
        return h

    @_quack_up_op.register_fake
    def _(xf, w_fc, rows, offs0, inv):
        return xf.new_empty(rows.numel(), w_fc.shape[1])

    @torch.library.custom_op("nanomoe::quack_up_bwd", mutates_args=())
    def _quack_up_bwd_op(dh: torch.Tensor, h: torch.Tensor, xf: torch.Tensor,
                         w_fc: torch.Tensor, rows: torch.Tensor, offs0: torch.Tensor,
                         inv: torch.Tensor) -> list[torch.Tensor]:
        E, DFF, C = w_fc.shape
        wb = w_fc.to(dh.dtype)
        ds = torch.empty_like(h)
        numel = ds.numel()
        _drelu_sq[(triton.cdiv(numel, 4096),)](dh.contiguous(), h, ds, numel, BLOCK=4096)
        # dx per pair: ds @ W1[e], ragged over experts; W1 is already (E, K=DFF, N=C)
        dx_pairs = _qk_gemm(ds, wb, cu_seqlens_m=offs0)
        dxf = torch.empty_like(xf)
        ones = torch.ones(rows.numel(), device=xf.device, dtype=xf.dtype)
        _combine_fwd[(dxf.shape[0], triton.cdiv(C, 256))](
            dx_pairs, ones, inv, dxf, C, K=inv.numel() // dxf.shape[0], BLOCK_C=256)
        # dW1[e] = ds_e^T @ xf[rows_e]: varlen along the reduction dim, gather fused on A
        dw = torch.empty(E, DFF, C, device=xf.device, dtype=torch.float32)
        _qk_gemm(xf.T, ds, out=dw.permute(0, 2, 1), cu_seqlens_k=offs0, A_idx=rows)
        return [dxf, dw]

    @_quack_up_bwd_op.register_fake
    def _(dh, h, xf, w_fc, rows, offs0, inv):
        return [torch.empty_like(xf), xf.new_empty(w_fc.shape, dtype=torch.float32)]

    def _quack_up_setup(ctx, inputs, output):
        xf, w_fc, rows, offs0, inv = inputs
        ctx.save_for_backward(xf, w_fc, rows, offs0, inv, output)

    def _quack_up_backward(ctx, dh):
        xf, w_fc, rows, offs0, inv, h = ctx.saved_tensors
        dxf, dw = torch.ops.nanomoe.quack_up_bwd(dh, h, xf, w_fc, rows, offs0, inv)
        return dxf, dw, None, None, None

    _quack_up_op.register_autograd(_quack_up_backward, setup_context=_quack_up_setup)


def fused_up(xf, w_fc, rows, offs, inv):
    """h = relu(xf[rows] @ w_fc[e].T)^2, grouped by expert. w_fc is (E, d_ff, C)."""
    # QuACK compiles per input shape, which is perfect for training (one fixed shape) and
    # pathological for benchmark eval (every batch a new shape, minutes of CuTe compilation
    # each). No-grad forwards take the shape-tolerant CUTLASS path instead.
    if HAS_QUACK and xf.is_cuda and torch.is_grad_enabled():
        offs0 = torch.cat([offs.new_zeros(1), offs])
        return torch.ops.nanomoe.quack_up(xf, w_fc, rows.to(torch.int32), offs0, inv)
    import torch.nn.functional as F
    xg = xf[rows]
    return F.relu(torch._grouped_mm(xg, w_fc.to(xg.dtype).transpose(1, 2), offs=offs)).square()


# -----------------------------------------------------------------------------
# Stage B, down half: y = h @ W2[e].T scaled by the gate and summed per token, with the
# whole backward owned explicitly instead of left to _grouped_mm's autograd.
#
# The forward is structurally unchanged (the scatter cannot move into the GEMM's store
# without atomics, and even sonic-moe keeps an expanded output plus a reduction kernel).
# The win is the backward: one kernel yields dy and dgate together, then dh and dW2 are
# two explicit QuACK GEMMs. dW2 uses the same varlen-K pattern as dW1, so no transposed
# copy of h is ever made.

if HAS_QUACK:

    @torch.library.custom_op("nanomoe::quack_down", mutates_args=())
    def _quack_down_op(h: torch.Tensor, w2: torch.Tensor, gate_o: torch.Tensor,
                       rows: torch.Tensor, offs0: torch.Tensor,
                       inv: torch.Tensor, n_tokens: int) -> list[torch.Tensor]:
        C = w2.shape[1]
        w2b = w2.to(h.dtype)
        # B is (E, K=DFF, N=C): the transpose view of the (E, C, DFF) storage order
        y = _qk_gemm(h, w2b.permute(0, 2, 1), cu_seqlens_m=offs0)
        out = torch.empty(n_tokens, C, device=h.device, dtype=h.dtype)
        _combine_fwd[(n_tokens, triton.cdiv(C, 256))](
            y, gate_o, inv, out, C, K=inv.numel() // n_tokens, BLOCK_C=256)
        return [out, y]

    @_quack_down_op.register_fake
    def _(h, w2, gate_o, rows, offs0, inv, n_tokens):
        return [h.new_empty(n_tokens, w2.shape[1]), h.new_empty(h.shape[0], w2.shape[1])]

    @torch.library.custom_op("nanomoe::quack_down_bwd", mutates_args=())
    def _quack_down_bwd_op(dout: torch.Tensor, y: torch.Tensor, h: torch.Tensor,
                           w2: torch.Tensor, gate_o: torch.Tensor, rows: torch.Tensor,
                           offs0: torch.Tensor) -> list[torch.Tensor]:
        E, C, DFF = w2.shape
        P = y.shape[0]
        w2b = w2.to(dout.dtype)
        # dy = gate * dout[rows] and dgate = <dout[rows], y>, one pass
        dy = torch.empty_like(y)
        dgate = torch.empty(P, device=y.device, dtype=torch.float32)
        _combine_bwd[(P,)](dout.contiguous(), y, gate_o, rows, dy, dgate, C, BLOCK_C=256)
        # dh per pair: dy @ W2[e]; W2 is already (E, K=C, N=DFF)
        dh = _qk_gemm(dy, w2b, cu_seqlens_m=offs0)
        # dW2[e] = dy_e^T @ h_e: varlen along the reduction dim, written in native layout
        dw2 = torch.empty(E, C, DFF, device=h.device, dtype=torch.float32)
        _qk_gemm(dy.T, h, out=dw2, cu_seqlens_k=offs0)
        return [dh, dw2, dgate.to(gate_o.dtype)]

    @_quack_down_bwd_op.register_fake
    def _(dout, y, h, w2, gate_o, rows, offs0):
        return [torch.empty_like(h), h.new_empty(w2.shape, dtype=torch.float32),
                torch.empty_like(gate_o)]

    def _quack_down_setup(ctx, inputs, output):
        h, w2, gate_o, rows, offs0, inv, n_tokens = inputs
        ctx.save_for_backward(h, w2, gate_o, rows, offs0, output[1])

    def _quack_down_backward(ctx, grads):
        dout, _ = grads
        h, w2, gate_o, rows, offs0, y = ctx.saved_tensors
        dh, dw2, dgate = torch.ops.nanomoe.quack_down_bwd(dout, y, h, w2, gate_o, rows, offs0)
        return dh, dw2, dgate, None, None, None, None

    _quack_down_op.register_autograd(_quack_down_backward, setup_context=_quack_down_setup)


def fused_down(h, w2, gate_o, rows, offs, inv, n_tokens):
    """out[t] = sum over token t's pairs of gate * (h @ w2[e].T). w2 is (E, C, d_ff)."""
    if HAS_QUACK and h.is_cuda and torch.is_grad_enabled():
        offs0 = torch.cat([offs.new_zeros(1), offs])
        out, _ = torch.ops.nanomoe.quack_down(h, w2, gate_o, rows.to(torch.int32),
                                              offs0, inv, n_tokens)
        return out
    y = torch._grouped_mm(h, w2.to(h.dtype).transpose(1, 2), offs=offs)
    out = torch.zeros(n_tokens, y.shape[1], device=y.device, dtype=y.dtype)
    out.index_add_(0, rows, y * gate_o.unsqueeze(-1))
    return out
