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

from min_gru import linear_scan
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
}


def aux_penalty(model):
    """Sum of grid-attraction penalties from the most recent forward."""
    total = 0.0
    for m in model.modules():
        pen = getattr(m, "_pen", None)
        if pen is not None:
            total = total + pen
    return total


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
    gen = torch.Generator().manual_seed(1 + seed)
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
