"""Experimental minGRU variants for the length-generalization loop.

Lab code: nothing here is wired into min_gru.py. Every variant keeps the
parallel associative-scan training path. Hypotheses, protocol, and the
round log live in EXPERIMENTS.md; each cell's result is appended to
lab_results.jsonl.

Usage (from the repo root or this directory):
       python experiments/variants.py TASK VARIANT [N_LAYERS]  # one cell
       python experiments/variants.py screen        # current round's cells
       TASK in {parity, S3}; VARIANT in VARIANTS below.
       MAX_STEPS env overrides the budget (default 1600);
       EXP_ROUND env tags results with the round number;
       EXP_CKPT=1 selects best checkpoint by val@T=128 (EXP_CKPT_T/_B
       override length/batches); EXP_NO_EARLYSTOP=1 runs the full budget.
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from min_gru import MinGRU, linear_scan
from probes import (
    BATCH,
    D_MODEL,
    EVAL_EVERY,
    LR,
    T_TRAIN,
    TASKS,
    GRUTagger,
    accuracy,
)

MAX_STEPS = int(os.environ.get("MAX_STEPS", 1600))
GEN_LENGTHS = (256, 512, 1024)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_results.jsonl")


# ---------------------------------------------------------------- mixers
class VariantSignedMinGRU(nn.Module):
    """SignedMinGRU with a pluggable eigenvalue head.

    h_t = a_t * h_{t-1} + z_t * Linear_h(x_t), linear-space scan.

    a_mode:
      coupled  : a = (1 - z) * tanh(s)   -- baseline SignedMinGRU
      tanh     : a = tanh(s)             -- H1: decoupled, open (-1, 1)
      hardtanh : a = hardtanh(s)         -- H2: decoupled, closed [-1, 1]
      ste-sign : a = sign(tanh(s)) via straight-through -- H3: |a| = 1
    """

    def __init__(self, input_size, hidden_size, a_mode="tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.a_mode = a_mode
        # Same construction order as SignedMinGRU so identical seeds give
        # identical weights (calibration cell must reproduce README).
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)
        self.linear_s = nn.Linear(input_size, hidden_size)

    def _coeffs(self, x):
        z = torch.sigmoid(self.linear_z(x))
        s = self.linear_s(x)
        if self.a_mode == "coupled":
            a = (1 - z) * torch.tanh(s)
        elif self.a_mode == "tanh":
            a = torch.tanh(s)
        elif self.a_mode == "hardtanh":
            a = F.hardtanh(s)
        elif self.a_mode == "ste-sign":
            t = torch.tanh(s)
            a = t + (torch.sign(t) - t).detach()
        else:
            raise ValueError(self.a_mode)
        return a, z * self.linear_h(x)

    def forward(self, x, h_0=None):
        if h_0 is None:
            h_0 = x.new_zeros(x.size(0), 1, self.hidden_size)
        a, b = self._coeffs(x)
        A, Bc = linear_scan(a, b)
        return A * h_0 + Bc

    def extra_repr(self):
        return f"a_mode={self.a_mode}"


def matrix_scan(M, b):
    """Hillis-Steele scan for h_t = M_t @ h_{t-1} + b_t with 2x2 blocks.

    M: (B, T, n, 2, 2) block transitions; b: (B, T, n, 2) inputs.
    Returns (A, Bc) with h_t = A_t @ h_0 + Bc_t. Same doubling scheme as
    linear_scan; identity blocks (not 1.0) pad the transition operand,
    and composition is A_current @ A_earlier (order matters now).
    """
    B_, T, n = M.shape[:3]
    eye = torch.eye(2, dtype=M.dtype, device=M.device)
    A, Bc = M, b
    offset = 1
    while offset < T:
        pad_A = eye.expand(B_, offset, n, 2, 2)
        A_prev = torch.cat([pad_A, A[:, : T - offset]], dim=1)
        B_prev = torch.cat(
            [b.new_zeros(B_, offset, n, 2), Bc[:, : T - offset]], dim=1
        )
        Bc = torch.einsum("btnij,btnj->btni", A, B_prev) + Bc
        A = A @ A_prev
        offset *= 2
    return A, Bc


class RotationMinGRU(nn.Module):
    """H4: non-abelian rung. 2x2 block transitions, parallel matrix scan.

    Hidden state is n = hidden_size/2 planar blocks. Per block:

        M_t = R(theta_t) @ diag(1, tanh(u_t))
        h_t = M_t @ h_{t-1} + z_t * Linear_h(x_t)

    Spectral norm of M is exactly 1 (a "hold" direction always exists);
    u -> +1 gives SO(2) rotations, u -> -1 gives O(2) reflections, and
    at theta = 0 the second channel reduces to signed-tanh. D3 (≅ S3)
    embeds in O(2), so one layer can represent the S3 running product.
    h_0 is a learnable generic vector per block (h_0 = 0 has no orbit
    under the group action; a vector on a reflection axis collapses
    reflections onto rotations).

    Lab code: parallel forward only, no step().
    """

    def __init__(self, input_size, hidden_size, snap=None, ortho=False, reg=0.0):
        super().__init__()
        assert hidden_size % 2 == 0, "rotation variant needs even hidden_size"
        self.hidden_size = hidden_size
        self.n_blocks = hidden_size // 2
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)
        self.linear_theta = nn.Linear(input_size, self.n_blocks)
        self.linear_u = nn.Linear(input_size, self.n_blocks)
        self.h0 = nn.Parameter(torch.randn(1, self.n_blocks, 2) * 0.5)
        # H10: snap theta to exact group angles 2*pi/K via straight-through
        # (per-block K cycled over `snap`). Rotations then have attractors
        # at group elements, like tanh's asymptote at -1: forward always
        # uses exact elements (zero angle drift at any length), gradients
        # flow through the soft angle.
        # H13: also snap the u channel to exact +/-1 via straight-through,
        # making every block an exact O(2) element (det +/-1). Removes the
        # tanh(u)^T decay that unsnapped reflections suffer; costs the
        # ability to forget (no contraction) — fine for state tracking,
        # wrong default for general sequence modeling.
        # H16: differentiable grid-attraction penalty. STE snap stays in
        # the forward pass; loss gains reg * mean((theta_soft - grid)^2),
        # which (a) shrinks the STE forward/backward mismatch driving
        # late-training collapse and (b) makes exact solutions attractors
        # of the LOSS, not just reachable points. u stays free (Round 5:
        # the forgetting path must remain).
        self.reg = reg
        self._pen = None  # set per forward when snap and reg > 0
        self.ortho = ortho
        self.snap = snap
        if snap is not None:
            self.register_buffer(
                "snap_step",
                torch.tensor(
                    [2 * math.pi / snap[j % len(snap)] for j in range(self.n_blocks)]
                ),
            )
        # Post-hoc analysis hooks (repair round 1); both default-off.
        # project_mask (n,) bool: snap tanh(u) -> +/-1 on masked blocks at
        # inference (theta is already snapped; this removes the remaining
        # reflection-magnitude inexactness). ablate_b_mask (n,) bool: zero
        # the injection b on masked blocks at inference.
        self.project_mask: torch.Tensor | None = None
        self.ablate_b_mask: torch.Tensor | None = None

    def _transitions(self, x):
        theta = self.linear_theta(x)                      # (B, T, n)
        if self.snap is not None:
            snapped = torch.round(theta / self.snap_step) * self.snap_step
            if self.reg > 0:
                self._pen = self.reg * ((theta - snapped.detach()) ** 2).mean()
            theta = theta + (snapped - theta).detach()
        c, s = torch.cos(theta), torch.sin(theta)
        d = torch.tanh(self.linear_u(x))
        if self.ortho:
            d = d + (torch.sign(d) - d).detach()
        elif self.project_mask is not None:
            d_exact = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
            d = torch.where(self.project_mask, d_exact, d)
        # R(theta) @ diag(1, d) = [[c, -s*d], [s, c*d]]
        row1 = torch.stack([c, -s * d], dim=-1)
        row2 = torch.stack([s, c * d], dim=-1)
        return torch.stack([row1, row2], dim=-2)          # (B, T, n, 2, 2)

    def forward(self, x, h_0=None):
        B, T, _ = x.shape
        z = torch.sigmoid(self.linear_z(x))
        b = (z * self.linear_h(x)).view(B, T, self.n_blocks, 2)
        if self.ablate_b_mask is not None:
            b = torch.where(self.ablate_b_mask[:, None], torch.zeros_like(b), b)
        M = self._transitions(x)
        A, Bc = matrix_scan(M, b)
        h0 = self.h0.expand(B, self.n_blocks, 2) if h_0 is None else h_0
        h = torch.einsum("btnij,bnj->btni", A, h0) + Bc
        return h.reshape(B, T, self.hidden_size)


class DeltaNetMixer(nn.Module):
    """Generalized-Householder delta rule (DeltaNet nh=1; DeltaProduct
    nh>1) -- the incumbent-mechanism comparison point for
    RotationMinGRU's snap variant, reimplemented from Yang et al. /
    Siems et al. as a plain-torch, sequential recurrence (no
    chunked/parallel form, no port of the official Triton kernels).

    Per head, state is a d_k x d_v matrix H (d_k = d_v = d_h / n_heads).
    Per token, nh micro-steps j = 1..nh apply IN ORDER:

        H <- (I - beta_j k_j k_j^T) H + beta_j k_j v_j^T

    with k_j L2-normalized and beta_j = 2*sigmoid(.) in (0, 2) (beta=2
    with unit k is an exact Householder reflection: I - 2 k k^T is
    orthogonal with det -1). Readout y_t = H_t^T q_t per head; heads
    concatenate through an output projection.

    nh=1 (DeltaNet) is the single-reflection floor -- their paper's own
    ablation axis. nh=2 (DeltaProduct) composes two reflections per
    token, which can realize a rotation: the rotation-capable
    counterpart to RotationMinGRU's snap grid.
    """

    def __init__(self, d_in, d_h, nh=1, n_heads=4, d_k=None, d_v=None):
        super().__init__()
        # d_k/d_v default to the original tied d_h//n_heads layout, so the
        # registered "deltanet"/"deltaproduct2" rows are construction-
        # identical to the recorded incumbent-delta evidence. Explicit
        # values decouple state size (n_heads * d_k * d_v) from d_h for
        # capacity-matched arms (e.g. n_heads=1, d_k=d_v=8 -> 64 state
        # elements, the promoted mixers' per-token state).
        if d_k is None:
            assert d_h % n_heads == 0, "d_h must be divisible by n_heads"
        self.d_in = d_in
        self.d_h = d_h
        self.nh = nh
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_h // n_heads
        self.d_v = d_v if d_v is not None else self.d_k
        self.linear_q = nn.Linear(d_in, n_heads * self.d_k)
        self.linear_k = nn.ModuleList(
            nn.Linear(d_in, n_heads * self.d_k) for _ in range(nh)
        )
        self.linear_v = nn.ModuleList(
            nn.Linear(d_in, n_heads * self.d_v) for _ in range(nh)
        )
        self.linear_beta = nn.ModuleList(nn.Linear(d_in, n_heads) for _ in range(nh))
        self.out_proj = nn.Linear(n_heads * self.d_v, d_h)

    def _step(self, H, k, v, beta):
        """One Householder micro-step: H <- (I - beta k k^T) H + beta k v^T.

        H: (B, n_heads, d_k, d_v); k: (B, n_heads, d_k) L2-normalized;
        v: (B, n_heads, d_v); beta: (B, n_heads).
        """
        kH = torch.einsum("bhk,bhkv->bhv", k, H)  # k^T H, per head
        beta = beta[..., None, None]
        H = H - beta * torch.einsum("bhk,bhv->bhkv", k, kH)
        H = H + beta * torch.einsum("bhk,bhv->bhkv", k, v)
        return H

    def forward(self, x):
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.n_heads, self.d_k)
        ks = [
            F.normalize(lin(x).view(B, T, self.n_heads, self.d_k), dim=-1)
            for lin in self.linear_k
        ]
        vs = [lin(x).view(B, T, self.n_heads, self.d_v) for lin in self.linear_v]
        betas = [2 * torch.sigmoid(lin(x)) for lin in self.linear_beta]  # (B,T,n_heads)
        H = x.new_zeros(B, self.n_heads, self.d_k, self.d_v)
        ys = []
        for t in range(T):
            for j in range(self.nh):
                H = self._step(H, ks[j][:, t], vs[j][:, t], betas[j][:, t])
            ys.append(torch.einsum("bhk,bhkv->bhv", q[:, t], H))
        y = torch.stack(ys, dim=1).reshape(B, T, self.n_heads * self.d_v)
        return self.out_proj(y)


# ---------------------------------------------------------------- blocks
class VariantBlock(nn.Module):
    """Pre-norm residual block around an arbitrary sequence mixer.

    Mirrors MinGRUBlock (LN -> mixer -> +x, LN -> MLP -> +x) but takes a
    mixer factory instead of a signed flag.
    """

    def __init__(self, d_model, mixer_factory, mlp_expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = mixer_factory(d_model, d_model)
        if mlp_expansion > 0:
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_expansion * d_model),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(mlp_expansion * d_model, d_model),
            )
        else:
            self.norm2 = None
            self.mlp = None

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class VariantStack(nn.Module):
    """Mirrors MinGRUStack: N VariantBlocks + final LN (in_proj identity)."""

    def __init__(self, d_model, n_layers, mixer_factory):
        super().__init__()
        self.blocks = nn.ModuleList(
            VariantBlock(d_model, mixer_factory) for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class VariantTagger(nn.Module):
    """Mirrors probes.MinGRUTagger: emb -> stack -> head."""

    def __init__(self, vocab, n_cls, mixer_factory, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.stack = VariantStack(D_MODEL, n_layers, mixer_factory)
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        return self.head(self.stack(self.emb(x)))


# -------------------------------------------------------------- registry
VARIANTS = {
    # Base minGRU (log-space, positive eigenvalues): the expressivity
    # floor quoted in the README. Imported from the promoted module
    # rather than re-implemented here.
    "log": lambda d_in, d_h: MinGRU(d_in, d_h),
    "signed-coupled": lambda d_in, d_h: VariantSignedMinGRU(d_in, d_h, "coupled"),
    "signed-tanh": lambda d_in, d_h: VariantSignedMinGRU(d_in, d_h, "tanh"),
    "signed-hardtanh": lambda d_in, d_h: VariantSignedMinGRU(d_in, d_h, "hardtanh"),
    "signed-ste": lambda d_in, d_h: VariantSignedMinGRU(d_in, d_h, "ste-sign"),
    "rotation": lambda d_in, d_h: RotationMinGRU(d_in, d_h),
    "rotation-snap": lambda d_in, d_h: RotationMinGRU(d_in, d_h, snap=(2, 3, 4, 6)),
    # K=1 blocks have theta = 0 always: pure diag(1, tanh) = signed-tanh
    # channel pairs. This grid unifies both winning mechanisms.
    "rotation-snap1": lambda d_in, d_h: RotationMinGRU(d_in, d_h, snap=(1, 2, 3, 4, 6)),
    # H13: every transition an exact O(2) element (theta snapped AND
    # u -> STE sign). Zero decay at any length for any learned solution.
    "ortho-snap": lambda d_in, d_h: RotationMinGRU(
        d_in, d_h, snap=(1, 2, 3, 4, 6), ortho=True
    ),
    # H16: STE snap + grid-attraction penalty at two strengths.
    "rotation-snap-reg01": lambda d_in, d_h: RotationMinGRU(
        d_in, d_h, snap=(2, 3, 4, 6), reg=0.1
    ),
    "rotation-snap-reg001": lambda d_in, d_h: RotationMinGRU(
        d_in, d_h, snap=(2, 3, 4, 6), reg=0.01
    ),
    # Incumbent-mechanism comparison (DeltaNet/DeltaProduct, Yang et al.
    # / Siems et al.): generalized-Householder delta rule reimplemented
    # in this harness (mechanism only -- official Triton/CUDA code is
    # not run here). nh=1 is the single-reflection floor (DeltaNet);
    # nh=2 composes two reflections per token, the rotation-capable
    # counterpart to rotation-snap.
    "deltanet": lambda d_in, d_h: DeltaNetMixer(d_in, d_h, nh=1),
    "deltaproduct2": lambda d_in, d_h: DeltaNetMixer(d_in, d_h, nh=2),
}


def aux_penalty(model):
    """Sum of grid-attraction penalties from the most recent forward."""
    total = 0.0
    for m in model.modules():
        pen = getattr(m, "_pen", None)
        if pen is not None:
            total = total + pen
    return total


def _deltanet_dense_construction(H0, ks, vs, betas):
    """Explicit dense construction of one token's delta-rule update,
    independent of DeltaNetMixer._step: builds each micro-step's
    Householder transition A_j = I - beta_j k_j k_j^T and injection
    b_j = beta_j k_j v_j^T from scratch, then composes

        H_nh = (A_nh ... A_1) H0 + sum_j (A_nh ... A_{j+1}) b_j

    which is the closed form of applying A_j H + b_j sequentially for
    j = 1..nh (order fixed by the spec). Used only to cross-check
    DeltaNetMixer._step's iterative application; not called in the
    forward path.
    """
    nh = len(ks)
    B, Hh, dk = ks[0].shape
    dv = vs[0].shape[-1]
    eye = torch.eye(dk).expand(B, Hh, dk, dk)
    As, bs = [], []
    for j in range(nh):
        kkT = torch.einsum("bhk,bhl->bhkl", ks[j], ks[j])
        As.append(eye - betas[j][..., None, None] * kkT)
        bs.append(betas[j][..., None, None] * torch.einsum("bhk,bhv->bhkv", ks[j], vs[j]))
    M = As[0]
    for j in range(1, nh):
        M = torch.einsum("bhkl,bhlm->bhkm", As[j], M)
    inj = torch.zeros(B, Hh, dk, dv)
    for j in range(nh):
        P = eye
        for later in range(j + 1, nh):
            P = torch.einsum("bhkl,bhlm->bhkm", As[later], P)
        inj = inj + torch.einsum("bhkl,bhlv->bhkv", P, bs[j])
    return torch.einsum("bhkl,bhlv->bhkv", M, H0) + inj


def _selftest_deltanet_dense_equivalence():
    """DeltaNetMixer._step applied iteratively (j = 1..nh, in order)
    must equal the explicit dense-matrix construction of the same
    update, pinning both the (I - beta k k^T) H + beta k v^T formula
    and the left-to-right micro-step composition order."""
    torch.manual_seed(1)
    B, Hh, dk, dv, nh = 2, 3, 4, 4, 3  # dv == dk: DeltaNetMixer requires d_v = d_k
    H0 = torch.randn(B, Hh, dk, dv)
    ks = [F.normalize(torch.randn(B, Hh, dk), dim=-1) for _ in range(nh)]
    vs = [torch.randn(B, Hh, dv) for _ in range(nh)]
    betas = [2 * torch.sigmoid(torch.randn(B, Hh)) for _ in range(nh)]

    mixer = DeltaNetMixer(d_in=1, d_h=Hh * dk, nh=nh, n_heads=Hh)
    H_iter = H0
    for j in range(nh):
        H_iter = mixer._step(H_iter, ks[j], vs[j], betas[j])
    H_dense = _deltanet_dense_construction(H0, ks, vs, betas)
    err = (H_iter - H_dense).abs().max().item()
    print(f"deltanet dense-matrix equivalence max abs diff: {err:.3e}")
    assert err < 1e-5, "delta-rule iterative update does not match dense construction"

    # order matters: swapping the last two micro-steps' (k, v, beta) must
    # change the result (guards against an order-insensitive/broken test).
    ks_swapped = ks[:-2] + [ks[-1], ks[-2]]
    vs_swapped = vs[:-2] + [vs[-1], vs[-2]]
    betas_swapped = betas[:-2] + [betas[-1], betas[-2]]
    H_dense_swapped = _deltanet_dense_construction(H0, ks_swapped, vs_swapped, betas_swapped)
    assert (H_dense - H_dense_swapped).abs().max().item() > 1e-4, (
        "dense construction is insensitive to micro-step order -- test is not pinning order"
    )


def _selftest_deltanet_householder():
    """beta=2, unit k: I - beta k k^T must be an exact Householder
    reflection -- orthogonal, det -1, eigenvalues {1 (x dk-1), -1}."""
    torch.manual_seed(2)
    dk = 6
    k = F.normalize(torch.randn(dk), dim=0)
    beta = torch.tensor(2.0)
    A = torch.eye(dk) - beta * torch.outer(k, k)
    orth_err = (A.T @ A - torch.eye(dk)).abs().max().item()
    print(f"deltanet Householder orthogonality max abs diff: {orth_err:.3e}")
    assert orth_err < 1e-5, "I - 2 k k^T is not orthogonal at beta=2, unit k"
    det = torch.linalg.det(A).item()
    assert abs(det - (-1.0)) < 1e-4, f"expected det -1, got {det:.4f}"
    eigvals = torch.linalg.eigvalsh(A).sort().values
    expected = torch.cat([torch.tensor([-1.0]), torch.full((dk - 1,), 1.0)])
    eig_err = (eigvals - expected).abs().max().item()
    assert eig_err < 1e-4, f"expected eigenvalues {{1 (x {dk - 1}), -1}}, got {eigvals.tolist()}"
    print("deltanet Householder algebra (orthogonal, det=-1, eigenvalues) ok")


def _selftest_deltanet_shapes_grad():
    """Shapes (B,T,d)->(B,T,d) and finite gradients on every parameter,
    for both registered nh values."""
    torch.manual_seed(3)
    B, T, d = 2, 9, 64
    x = torch.randn(B, T, d)
    for nh in (1, 2):
        mixer = DeltaNetMixer(d, d, nh=nh, n_heads=4)
        y = mixer(x)
        assert y.shape == (B, T, d), f"nh={nh}: expected shape {(B, T, d)}, got {tuple(y.shape)}"
        y.sum().backward()
        for name, p in mixer.named_parameters():
            assert p.grad is not None, f"nh={nh}: {name} received no gradient"
            assert torch.isfinite(p.grad).all(), f"nh={nh}: {name} has non-finite gradient"
    print("deltanet shapes/gradients ok (nh=1, nh=2)")


def selftest():
    """matrix_scan vs brute-force sequential recurrence."""
    torch.manual_seed(0)
    B, T, n = 3, 17, 5
    M = torch.randn(B, T, n, 2, 2) * 0.5
    b = torch.randn(B, T, n, 2)
    h0 = torch.randn(B, n, 2)
    A, Bc = matrix_scan(M, b)
    h_scan = torch.einsum("btnij,bnj->btni", A, h0) + Bc
    h = h0
    hs = []
    for t in range(T):
        h = torch.einsum("bnij,bnj->bni", M[:, t], h) + b[:, t]
        hs.append(h)
    h_seq = torch.stack(hs, dim=1)
    err = (h_scan - h_seq).abs().max().item()
    print(f"matrix_scan vs sequential max abs diff: {err:.3e}")
    assert err < 1e-4, "matrix_scan does not match sequential recurrence"
    # gradients flow
    M.requires_grad_(True)
    A, Bc = matrix_scan(M, b)
    Bc.sum().backward()
    assert M.grad is not None and torch.isfinite(M.grad).all()
    print("matrix_scan gradcheck-lite ok")
    # delta-rule (DeltaNet/DeltaProduct) checks -- spec section 10.
    _selftest_deltanet_dense_equivalence()
    _selftest_deltanet_householder()
    _selftest_deltanet_shapes_grad()


# ---------------------------------------------------------------- driver
def run_cell(task, variant, n_layers=1, seed=0, max_steps=None, return_model=False):
    """One (task, variant, layers, seed) cell under the probes.py protocol,
    with extra generalization lengths. Appends a JSON line to RESULTS.
    seed=0 matches probes.py exactly; other seeds vary init and data
    order (eval seeds stay fixed for comparability). variant="gru" runs
    the probes.py GRU baseline through this harness (same protocol,
    extended eval lengths)."""
    if max_steps is None:
        max_steps = MAX_STEPS
    make, vocab, n_cls = TASKS[task]
    torch.manual_seed(seed)
    # seed=0 reproduces the legacy manual_seed(1) exactly (1 + 10_000 * 0 ==
    # 1); other seeds vary init and data order. The 10_000 multiplier keeps
    # seed>=1's train generator (10001, 20001, ...) far from the small fixed
    # eval seeds (2/3/4/5) used below -- plain `1 + seed` put seed=1's train
    # gen at 2 (colliding with the early-stop eval seed) and seed=2's at 3
    # (colliding with the acc_in eval seed), silently overlapping train and
    # eval examples at T_TRAIN for those seeds. Eval seeds themselves are
    # untouched. Mirrors probes.py's run_one fix for the identical flaw.
    gen = torch.Generator().manual_seed(1 + 10_000 * seed)
    if variant == "gru":
        model = GRUTagger(vocab, n_cls, n_layers)
    else:
        model = VariantTagger(vocab, n_cls, VARIANTS[variant], n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0, steps_used = time.time(), max_steps
    # H15 (EXP_CKPT=1): the exact solution is reachable but not a stable
    # attractor of training (runs wander in and out of it — Round 6).
    # Select the best checkpoint by validation at T=128, a length not in
    # the test set (256/512/1024), and run the full budget.
    ckpt_select = bool(os.environ.get("EXP_CKPT"))
    val_T = int(os.environ.get("EXP_CKPT_T", 128))
    val_B = int(os.environ.get("EXP_CKPT_B", 2))
    best_val, best_state, best_step = -1.0, None, 0
    for step in range(1, max_steps + 1):
        x, y = make(BATCH, T_TRAIN, gen)
        loss = F.cross_entropy(model(x).reshape(-1, n_cls), y.reshape(-1))
        loss = loss + aux_penalty(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            if ckpt_select:
                val = accuracy(model, make, val_T, seed=5, n_batches=val_B)
                if val > best_val:
                    best_val, best_step = val, step
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
            elif not os.environ.get("EXP_NO_EARLYSTOP"):
                if accuracy(model, make, T_TRAIN, seed=2, n_batches=2) >= 0.999:
                    steps_used = step
                    break
    if ckpt_select and best_state is not None:
        model.load_state_dict(best_state)
        steps_used = best_step
    accs = {str(T_TRAIN): round(accuracy(model, make, T_TRAIN, seed=3), 4)}
    for T in GEN_LENGTHS:
        accs[str(T)] = round(accuracy(model, make, T, seed=4), 4)
    rec = {
        "round": os.environ.get("EXP_ROUND", "?"),
        "task": task,
        "variant": variant,
        "layers": n_layers,
        "seed": seed,
        "steps": steps_used,
        "acc": accs,
        "secs": round(time.time() - t0, 1),
        "max_steps": max_steps,
    }
    if ckpt_select:
        rec["ckpt"] = {"step": best_step, "val128": round(best_val, 4)}
    print(json.dumps(rec), flush=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if return_model:
        return rec, model
    return rec


# Current round's screening cells; edited per round (see EXPERIMENTS.md).
# Post-loop verification (Rec 3): every baseline row quoted in the
# synthesis re-run in the current environment, 3 seeds, original
# early-stop protocol (no EXP_CKPT). Seed-0 rows already in
# lab_results.jsonl from Rounds 1-2 are not repeated.
SCREEN = [
    ("parity", "signed-coupled", 1, 1),
    ("parity", "signed-coupled", 1, 2),
    ("S3", "signed-coupled", 1, 0),
    ("S3", "signed-coupled", 1, 1),
    ("S3", "signed-coupled", 1, 2),
    ("S3", "signed-coupled", 4, 1),
    ("S3", "signed-coupled", 4, 2),
    ("parity", "gru", 1, 0),
    ("parity", "gru", 1, 1),
    ("parity", "gru", 1, 2),
    ("S3", "gru", 1, 0),
    ("S3", "gru", 1, 1),
    ("S3", "gru", 1, 2),
]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "screen":
        for cell in SCREEN:
            run_cell(*cell)
    else:
        run_cell(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
