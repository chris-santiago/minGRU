"""Hetero-stack training-fix lab: intervention arms for the S3-hier
`signed -> rotation` trainability problem (hetero-loop rounds).

Baseline pathology (rounds hetero-legB-v2 / hetero-legB-ceiling):
minGRU-hetero-sr on S3-hier sits near chance at the 1600-step budget and
reaches the exact composer on 1/3 seeds at 6400; the winner then length-
generalizes near-perfectly. The research synthesis
(.claude/output/research/2026-07-13-hetero-stack-hypotheses.md) ranks
training-side fixes; this driver implements them as composable flags on
top of an exact replication of the probes.py protocol, so every arm is
matched-seed comparable to the recorded baselines (flags all off =
bit-identical RNG path to probes.run_one CKPT rows).

Interventions (all default off):
  --soft-warmup N   composer snap disabled (continuous angles) for the
                    first N steps  [Tier-1 #1, Guo et al. 2021]
  --soft-blend B    after the warmup, blend snap in linearly over B steps
                    (STE forward = theta + alpha*(snapped-theta),
                    alpha: 0 -> 1)
  --commit-lambda L ramped commitment penalty L*mean((theta - sg(theta_snap))^2)
    --commit-ramp R on the composer's angle head, lambda ramped 0 -> L over
                    R steps (default 800)  [Tier-1 #3, Yin/ProxQuant/Nagel]
  --grad-noise ETA  annealed additive gradient noise, sigma^2 =
                    ETA/(1+t)^0.55  [Tier-1 #4, Neelakantan et al.]
  --identity-warmup N  composer transition frozen at the identity for the
                    first N steps (injection still learns)  [Tier-1 #6]
  --curriculum P:MAX  per-step, with prob P train length is an even draw
                    from [2, MAX] instead of T_TRAIN  [Tier-1 #2, Zaremba
                    rule approximated per-batch; at T=2 the running
                    composition IS the generator]

Protocol: CKPT best-val@128 (seed 5, 2 batches) over the full budget,
eval acc@64 (seed 3) and 256/512/1024 (seed 4), matching
experiments/variants.py run_cell. Appends a row per run to
lab_results.jsonl with the intervention config recorded.

Usage:
  uv run --python 3.12 --with 'torch==2.5.1' python experiments/hetero_lab.py \
      --round hetero-loop-02-softwarmup --seed 0 --steps 1600 \
      --soft-warmup 400 --soft-blend 100
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from min_gru import RotationMinGRU
from probes import (
    BATCH, CKPT_T, D_MODEL, EVAL_EVERY, LR, T_TRAIN, TASKS, accuracy, build,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import VARIANTS, VariantBlock  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_results.jsonl")
GEN_LENGTHS = (256, 512, 1024)


class _LabRotation(RotationMinGRU):
    """RotationMinGRU with lab knobs. Instances are created by reassigning
    __class__ on an already-built RotationMinGRU (parameters/buffers and
    their RNG-matched init are untouched; only method resolution changes),
    so arm models stay init-identical to the recorded baseline seeds.

    Knobs (class-level defaults keep flag-off behavior bit-identical):
      snap_alpha    : 1.0 = full STE snap (parent behavior); 0.0 = soft
                      (continuous angles); in between = linear blend of
                      the STE correction.
      identity_mode : True forces the per-block transition M to the
                      identity (theta=0, diag(1,1)); injection unchanged.
    """

    snap_alpha = 1.0
    identity_mode = False
    _capture_theta = False  # when True, stash (theta_soft, theta_used)

    def _coeffs(self, x, dt=None):
        theta = self.linear_theta(x)
        theta_soft = theta
        if self.snap is not None and self.snap_alpha > 0.0:
            snapped = torch.round(theta / self.snap_step) * self.snap_step
            theta = theta + self.snap_alpha * (snapped - theta).detach()
        if self._capture_theta:
            self._theta_soft, self._theta_used = theta_soft, theta
        if self.identity_mode:
            zeros = torch.zeros_like(theta)
            cos_t, sin_t = torch.ones_like(theta), zeros
            d = torch.ones_like(theta)
        else:
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            d = torch.tanh(self.linear_u(x))
        row0 = torch.stack([cos_t, -sin_t * d], dim=-1)
        row1 = torch.stack([sin_t, cos_t * d], dim=-1)
        M = torch.stack([row0, row1], dim=-2)
        if dt is not None:
            M = self._decay_gamma(dt).unsqueeze(-1).unsqueeze(-1) * M
        z = torch.sigmoid(self.linear_z(x))
        b = z * self.linear_h(x)
        b = b.reshape(*b.shape[:-1], self.n_blocks, 2)
        return M, b


# Hetero stacks over experiments/variants.py building blocks (Tier-2 #7:
# swap the snapped-rotation composer for the DeltaProduct nh=2 mixer,
# which solved plain S3 with no snap in round incumbent-delta). Built
# from the same VariantBlock/final-LN shape as VariantStack, so rows are
# protocol-comparable to both the hetero-sr rows and incumbent-delta.
HETERO_FACTORY_MODELS = {
    "hetero-sd2": ("signed-tanh", "deltaproduct2"),
    "hetero-d2s": ("deltaproduct2", "signed-tanh"),
}


class _ListVariantStack(nn.Module):
    """VariantStack with a per-layer factory tuple instead of one factory."""

    def __init__(self, d_model, factory_names):
        super().__init__()
        self.blocks = nn.ModuleList(
            VariantBlock(d_model, VARIANTS[name]) for name in factory_names
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class _HeteroVariantTagger(nn.Module):
    def __init__(self, vocab, n_cls, factory_names):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.stack = _ListVariantStack(D_MODEL, factory_names)
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        return self.head(self.stack(self.emb(x)))


class VQBottleneck(nn.Module):
    """K-code vector quantizer on the residual stream, STE pass-through.

    Motivation (loop 11): the composer's per-instance maps vary more
    within a generator class than across classes -- the composer sees a
    continuous feature cloud where plain S3's composer sees 6 discrete
    embeddings. Quantizing the interface collapses within-class map
    variance structurally.

    Training penalty (VQ-VAE codebook + commitment terms) is stashed on
    ``self._pen`` during training forwards; the driver adds it to the
    loss (same collection convention as variants.aux_penalty). Eval
    forwards quantize identically (deploy mode IS quantized) but skip
    the penalty.
    """

    def __init__(self, d_model, n_codes=8, commit=0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, d_model))
        self.commit = commit
        self._pen = None

    def forward(self, x):
        d2 = (
            x.pow(2).sum(-1, keepdim=True)
            - 2 * x @ self.codebook.T
            + self.codebook.pow(2).sum(-1)
        )
        e = self.codebook[d2.argmin(-1)]
        if self.training:
            self._pen = (x.detach() - e).pow(2).mean() + self.commit * (
                x - e.detach()
            ).pow(2).mean()
        return x + (e - x).detach()


class _VQHeteroTagger(nn.Module):
    """_HeteroVariantTagger with a VQBottleneck between the two blocks."""

    def __init__(self, vocab, n_cls, factory_names, n_codes=8):
        super().__init__()
        assert len(factory_names) == 2, "VQ interface expects a 2-block stack"
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.stack = _ListVariantStack(D_MODEL, factory_names)
        self.vq = VQBottleneck(D_MODEL, n_codes=n_codes)
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        x = self.emb(x)
        x = self.stack.blocks[0](x)
        x = self.vq(x)
        x = self.stack.blocks[1](x)
        return self.head(self.stack.norm_out(x))


def _rotation_mixers(model):
    mixers = []
    for blk in model.stack.blocks:
        m = getattr(blk, "mingru", None) or getattr(blk, "mixer", None)
        if isinstance(m, RotationMinGRU):
            mixers.append(m)
    return mixers


def _parse_curriculum(spec):
    """'P:MAX' -> (float prob in [0,1], even int max >= 2); '' -> (0.0, 0)."""
    if not spec:
        return 0.0, 0
    try:
        p_str, max_str = spec.split(":")
        p, mx = float(p_str), int(max_str)
    except ValueError as e:
        raise SystemExit(f"--curriculum expects P:MAX (e.g. 0.25:16), got {spec!r}") from e
    if not (0.0 <= p <= 1.0) or mx < 2 or mx % 2 != 0:
        raise SystemExit(
            f"--curriculum P:MAX needs 0<=P<=1 and even MAX>=2, got {spec!r}"
        )
    return p, mx


def _hard_mode_eval(model, rot, make, T, seed, n_batches):
    """accuracy() with every rotation mixer forced to hard/deployment
    mode (full snap, no identity freeze), schedules restored afterward.

    Checkpoint selection must score the DEPLOYABLE model: during a soft
    or identity warmup the training-mode model can post val scores its
    snapped counterpart cannot reproduce, and selecting on those keeps a
    checkpoint that collapses at the final (hard-mode) eval.
    """
    saved = [(m.snap_alpha, m.identity_mode) for m in rot]
    for m in rot:
        m.snap_alpha, m.identity_mode = 1.0, False
    acc = accuracy(model, make, T, seed=seed, n_batches=n_batches)
    for m, (a, i) in zip(rot, saved):
        m.snap_alpha, m.identity_mode = a, i
    return acc


def run_arm(args):
    args.curriculum_p, args.curriculum_max = _parse_curriculum(args.curriculum)
    if args.ckpt_t in GEN_LENGTHS or args.ckpt_t <= 0:
        raise SystemExit(
            f"--ckpt-t {args.ckpt_t} is a reported test length (or invalid); "
            f"selection must not leak into the metric. Test lengths: {GEN_LENGTHS}"
        )
    make, vocab, n_cls = TASKS[args.task]
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(1 + 10_000 * args.seed)
    if args.model == "hetero-svq8-d2":
        model = _VQHeteroTagger(
            vocab, n_cls, HETERO_FACTORY_MODELS["hetero-sd2"], n_codes=8
        )
    elif args.model in HETERO_FACTORY_MODELS:
        model = _HeteroVariantTagger(vocab, n_cls, HETERO_FACTORY_MODELS[args.model])
    else:
        model = build(args.task, args.model, vocab, n_cls, None)
    rot = _rotation_mixers(model)
    for m in rot:
        m.__class__ = _LabRotation
    rotation_flags = (
        args.soft_warmup or args.soft_blend or args.identity_warmup
        or args.commit_lambda > 0.0
    )
    if rotation_flags and not rot:
        raise ValueError(
            f"model {args.model!r} has no rotation mixer; composer-side "
            "flags (--soft-warmup/--soft-blend/--identity-warmup/"
            "--commit-lambda) do not apply"
        )
    if (args.commit_lambda > 0.0 or args.soft_warmup or args.soft_blend) and any(
        m.snap is None for m in rot
    ):
        raise ValueError(
            "--commit-lambda/--soft-warmup need a snap grid, but a rotation "
            "mixer has snap=None (continuous angles)"
        )
    # dedicated generators so interventions never perturb the train-data
    # or init RNG stream (keeps arms matched-seed comparable)
    noise_gen = torch.Generator().manual_seed(90_001 + args.seed)
    curr_gen = torch.Generator().manual_seed(80_001 + args.seed)

    n_cur_short = 0
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    best_val, best_state, best_step = -1.0, None, 0
    for step in range(1, args.steps + 1):
        # per-step knob schedule
        if args.soft_warmup or args.soft_blend:
            hard_at = args.soft_warmup + args.soft_blend
            if step <= args.soft_warmup:
                alpha = 0.0
            elif step <= hard_at:
                alpha = (step - args.soft_warmup) / max(args.soft_blend, 1)
            else:
                alpha = 1.0
            for m in rot:
                m.snap_alpha = alpha
        if args.identity_warmup:
            for m in rot:
                m.identity_mode = step <= args.identity_warmup

        T = T_TRAIN
        if args.curriculum_p > 0.0:
            if torch.rand((), generator=curr_gen).item() < args.curriculum_p:
                T = 2 * int(
                    torch.randint(1, args.curriculum_max // 2 + 1, (), generator=curr_gen)
                )
                n_cur_short += 1
        x, y = make(BATCH, T, gen)
        if args.commit_lambda > 0.0:
            for m in rot:
                m._capture_theta = True
        loss = F.cross_entropy(model(x).reshape(-1, n_cls), y.reshape(-1))
        # VQ-VAE codebook/commitment penalties from this forward, if any
        # (VQBottleneck._pen; same collection idea as variants.aux_penalty)
        for m in model.modules():
            pen = getattr(m, "_pen", None)
            if pen is not None:
                loss = loss + pen
                m._pen = None
        if args.commit_lambda > 0.0:
            lam = args.commit_lambda * min(1.0, step / max(args.commit_ramp, 1))
            for m in rot:
                snapped = torch.round(m._theta_soft.detach() / m.snap_step) * m.snap_step
                loss = loss + lam * (m._theta_soft - snapped).pow(2).mean()
                m._capture_theta = False
                m._theta_soft = m._theta_used = None
        opt.zero_grad()
        loss.backward()
        if args.grad_noise > 0.0:
            # Neelakantan-style annealed GRADIENT noise: added to p.grad
            # before opt.step() so it passes through Adam's preconditioner
            # (post-step weight noise would be a different intervention).
            sigma = (args.grad_noise / (1 + step) ** 0.55) ** 0.5
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn(p.shape, generator=noise_gen) * sigma)
        opt.step()
        if step % EVAL_EVERY == 0:
            val = _hard_mode_eval(model, rot, make, args.ckpt_t, seed=5, n_batches=2)
            if val > best_val:
                best_val, best_step = val, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Evaluate in hard/deployment mode regardless of where the schedules
    # ended (guards warmup horizons >= --steps from producing soft-mode rows).
    for m in rot:
        m.snap_alpha, m.identity_mode = 1.0, False
    if best_state is not None:
        model.load_state_dict(best_state)
    accs = {str(T_TRAIN): round(accuracy(model, make, T_TRAIN, seed=3), 4)}
    for T in GEN_LENGTHS:
        accs[str(T)] = round(accuracy(model, make, T, seed=4), 4)
    config = {
        k: v
        for k, v in {
            "soft_warmup": args.soft_warmup,
            "soft_blend": args.soft_blend,
            "commit_lambda": args.commit_lambda,
            "commit_ramp": args.commit_ramp if args.commit_lambda else 0,
            "grad_noise": args.grad_noise,
            "identity_warmup": args.identity_warmup,
            "curriculum_p": args.curriculum_p,
            "curriculum_max": args.curriculum_max if args.curriculum_p else 0,
            "ckpt_t": args.ckpt_t if args.ckpt_t != CKPT_T else 0,
        }.items()
        if v
    }
    if args.curriculum_p > 0.0:
        config["n_short_batches"] = n_cur_short
    rec = {
        "round": args.round,
        "task": args.task,
        "variant": args.model,
        "layers": len(model.stack.blocks),
        "seed": args.seed,
        "steps": best_step,
        "acc": accs,
        "secs": round(time.time() - t0, 1),
        "max_steps": args.steps,
        "ckpt": {"step": best_step, f"val{args.ckpt_t}": round(best_val, 4)},
        "config": config,
    }
    print(json.dumps(rec), flush=True)
    if not args.dry_run:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round", required=True)
    p.add_argument("--task", default="S3-hier")
    p.add_argument("--model", default="minGRU-hetero-sr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--soft-warmup", type=int, default=0)
    p.add_argument("--soft-blend", type=int, default=0)
    p.add_argument("--commit-lambda", type=float, default=0.0)
    p.add_argument("--commit-ramp", type=int, default=800)
    p.add_argument("--grad-noise", type=float, default=0.0)
    p.add_argument("--identity-warmup", type=int, default=0)
    p.add_argument(
        "--curriculum", default="",
        help="P:MAX, e.g. 0.25:16 -- prob P of an even short length in [2, MAX]",
    )
    p.add_argument(
        "--ckpt-t", type=int, default=CKPT_T,
        help="checkpoint-selection eval length (default CKPT_T=128). Must "
        "not be one of the reported test lengths (256/512/1024) or "
        "selection leaks into the metric; use when val@128 saturates and "
        "cannot discriminate (e.g. 384 for gen-consolidation arms).",
    )
    p.add_argument("--dry-run", action="store_true", help="print row, skip ledger append")
    run_arm(p.parse_args())


if __name__ == "__main__":
    main()
