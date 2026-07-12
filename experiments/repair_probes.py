"""Repair round 1: inference-time projection + injection probes.

Tests the "failure is calibration, not search" hypothesis directly:

(a) Projection: on failed rotation-snap seeds (s2, s6), identify the
    near-homomorphism blocks from weights and snap their tanh(u) to
    exact +/-1 at inference (theta is already snapped). Zero training.
    If accuracy at long lengths recovers, the residual seed variance
    was pure reflection-magnitude calibration. Strategies compared:
    project the best block only / all near blocks (hom err < 0.1) /
    all 32 blocks. Control: s0 (already exact) must not degrade.

(b) Probe gap: per-block mean injection norm ||b|| (not just the gate
    z) on the automaton vs other blocks, plus injection ablation on
    the automaton blocks of perfect seeds — does the exact solution
    actually depend on its open injection stream?

Writes repair_results.json. Retrains its cells (winners' protocol);
models are not persisted anywhere.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes import COMPOSE, TASKS, accuracy
from mechanism_probes import token_transitions
from variants import GEN_LENGTHS, run_cell

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repair_results.json")
NEAR_HOM = 0.1


def block_hom_errors(model):
    """(n,) worst-pair Frobenius hom error per block; plus M and per-block
    mean injection norm ||b|| over the 6 tokens."""
    block, xn, z = token_transitions(model, 6)
    with torch.no_grad():
        M = block.mixer._transitions(xn.unsqueeze(1))[:, 0]     # (6, n, 2, 2)
        n = M.shape[1]
        err = torch.zeros(n)
        for g in range(6):
            for h in range(6):
                gh = COMPOSE[g, h].item()
                diff = M[g] @ M[h] - M[gh]
                err = torch.maximum(err, diff.flatten(1).norm(dim=1))
        b = (z * block.mixer.linear_h(xn)).view(6, n, 2)
        b_norm = b.norm(dim=-1).mean(dim=0)                     # (n,)
    return err, b_norm


def eval_cell(model, task):
    make = TASKS[task][0]
    accs = {"64": round(accuracy(model, make, 64, seed=3), 4)}
    for T in GEN_LENGTHS:
        accs[str(T)] = round(accuracy(model, make, T, seed=4), 4)
    return accs


def with_project_mask(model, mask, task):
    mixer = model.stack.blocks[0].mixer
    mixer.project_mask = mask
    accs = eval_cell(model, task)
    mixer.project_mask = None
    return accs


def with_ablate_mask(model, mask, task):
    mixer = model.stack.blocks[0].mixer
    mixer.ablate_b_mask = mask
    accs = eval_cell(model, task)
    mixer.ablate_b_mask = None
    return accs


def main():
    os.environ["EXP_CKPT"] = "1"
    os.environ["EXP_ROUND"] = "repair-1"
    results = {}

    # (a) projection on failed seeds; s0 as no-harm control
    for seed in (0, 2, 6):
        rec, model = run_cell("S3", "rotation-snap", 1, seed, return_model=True)
        err, b_norm = block_hom_errors(model)
        best = int(err.argmin())
        near = err < NEAR_HOM
        best_mask = torch.zeros_like(near)
        best_mask[best] = True
        entry = {
            "baseline": rec["acc"],
            "hom_err_best": round(err[best].item(), 6),
            "near_blocks": int(near.sum()),
            "project_best": with_project_mask(model, best_mask, "S3"),
            "project_near": with_project_mask(model, near, "S3"),
            "project_all": with_project_mask(model, torch.ones_like(near), "S3"),
            # (b) ||b|| context: automaton block vs the rest
            "b_norm_best_block": round(b_norm[best].item(), 4),
            "b_norm_median": round(b_norm.median().item(), 4),
        }
        # (b) injection ablation on the near-hom (automaton) blocks
        entry["ablate_b_near"] = with_ablate_mask(model, near, "S3")
        results[f"s{seed}"] = entry
        print(json.dumps({f"s{seed}": entry}), flush=True)

    # (b) same probes on the other perfect seed (s1, z=0.70 puzzle)
    rec, model = run_cell("S3", "rotation-snap", 1, 1, return_model=True)
    err, b_norm = block_hom_errors(model)
    near = err < NEAR_HOM
    results["s1"] = {
        "baseline": rec["acc"],
        "hom_err_best": round(err.min().item(), 6),
        "near_blocks": int(near.sum()),
        "b_norm_best_block": round(b_norm[int(err.argmin())].item(), 4),
        "b_norm_median": round(b_norm.median().item(), 4),
        "ablate_b_near": with_ablate_mask(model, near, "S3"),
        "project_near": with_project_mask(model, near, "S3"),
    }
    print(json.dumps({"s1": results["s1"]}), flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
