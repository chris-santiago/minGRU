"""Repair round 2: (b-lite) project + fine-tune, and (c) hom-error selection.

Part A — b-lite: on failed seeds (s2, s6), apply the O(2) projection
mask to the identified near-hom blocks, then fine-tune ~400 steps so
the readout can re-learn to rely on the now-exact features (round-1
result: projection alone recovers only +3..8 pts @1024 because the
residual failure is readout-side). Best-by-val@128 state within the
fine-tune is kept. Also evaluates that state with the mask removed —
did fine-tuning internalize the exactness into the u weights?

Part B — c: re-run the failed seeds' original training while tracking
BOTH selection criteria every 100 steps: val@128 (current protocol)
and raw best-block homomorphism error (data-free, computed from
weights). Selects a checkpoint under each and compares accuracy at
all lengths. Answers: does the certificate beat the held-out length
as a selection rule?

Writes repair2_results.json.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes import BATCH, EVAL_EVERY, LR, T_TRAIN, TASKS, accuracy
from repair_probes import block_hom_errors, eval_cell, NEAR_HOM
from variants import VARIANTS, VariantTagger, run_cell

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repair2_results.json")
FT_STEPS = 400
FT_EVAL_EVERY = 50


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def finetune_projected(seed):
    """Part A for one seed."""
    rec, model = run_cell("S3", "rotation-snap", 1, seed, return_model=True)
    err, _ = block_hom_errors(model)
    near = err < NEAR_HOM
    mixer = model.stack.blocks[0].mixer
    mixer.project_mask = near

    make, _, n_cls = TASKS["S3"]
    gen = torch.Generator().manual_seed(5000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, best_step = -1.0, None, 0
    for step in range(1, FT_STEPS + 1):
        x, y = make(BATCH, T_TRAIN, gen)
        loss = F.cross_entropy(model(x).reshape(-1, n_cls), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % FT_EVAL_EVERY == 0:
            val = accuracy(model, make, 128, seed=5, n_batches=2)
            if val > best_val:
                best_val, best_state, best_step = val, snapshot(model), step
    model.load_state_dict(best_state)

    with_mask = eval_cell(model, "S3")
    mixer.project_mask = None
    without_mask = eval_cell(model, "S3")
    return {
        "baseline": rec["acc"],
        "near_blocks": int(near.sum()),
        "ft_best_val128": round(best_val, 4),
        "ft_best_step": best_step,
        "after_ft_with_mask": with_mask,
        "after_ft_without_mask": without_mask,
    }


def dual_criterion_training(seed):
    """Part B for one seed: track val@128 and raw hom error; select both."""
    make, vocab, n_cls = TASKS["S3"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1 + seed)
    model = VariantTagger(vocab, n_cls, VARIANTS["rotation-snap"], 1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, val_state, val_step = -1.0, None, 0
    best_hom, hom_state, hom_step = float("inf"), None, 0
    for step in range(1, 1601):
        x, y = make(BATCH, T_TRAIN, gen)
        loss = F.cross_entropy(model(x).reshape(-1, n_cls), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            val = accuracy(model, make, 128, seed=5, n_batches=2)
            if val > best_val:
                best_val, val_state, val_step = val, snapshot(model), step
            err, _ = block_hom_errors(model)
            hom = err.min().item()
            if hom < best_hom:
                best_hom, hom_state, hom_step = hom, snapshot(model), step
    out = {}
    model.load_state_dict(val_state)
    out["val_select"] = {
        "step": val_step,
        "val128": round(best_val, 4),
        "acc": eval_cell(model, "S3"),
    }
    model.load_state_dict(hom_state)
    out["hom_select"] = {
        "step": hom_step,
        "hom_err": round(best_hom, 6),
        "acc": eval_cell(model, "S3"),
    }
    return out


def main():
    os.environ["EXP_CKPT"] = "1"
    os.environ["EXP_ROUND"] = "repair-2"
    results = {}
    for seed in (2, 6):
        results[f"s{seed}_ft"] = finetune_projected(seed)
        print(json.dumps({f"s{seed}_ft": results[f"s{seed}_ft"]}), flush=True)
    for seed in (2, 6):
        results[f"s{seed}_select"] = dual_criterion_training(seed)
        print(json.dumps({f"s{seed}_select": results[f"s{seed}_select"]}), flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
