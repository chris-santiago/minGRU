"""Mechanism verification probes (post-loop, Rec 2).

Converts the loop's headline from inference to demonstration:

1. Homomorphism test (S3, rotation-snap): extract each token's learned
   2x2 transition M(g) per block and check M(g) @ M(h) ~= M(g o h)
   against the S3 composition table. If the model learned the
   automaton, at least one block must be a (near-)faithful D3
   representation with its injection gate z ~= 0.
2. Eigenvalue histogram (parity, signed-tanh): per-channel transition
   coefficients a(token). The automaton channels must sit at
   (a(0), a(1)) ~= (+1, -1): hold on 0, flip on 1.

Trains the probed cells fresh (the loop kept no checkpoints; winners
solve in 100-300 steps) under the same protocol as their headline
rows, then inspects the trained weights. Transitions are per-token
constants here: tokens enter through an embedding and one LayerNorm,
with no context dependence before the mixer.

Usage: python experiments/mechanism_probes.py
Writes a summary to experiments/mechanism_results.json.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes import COMPOSE
from variants import run_cell

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mechanism_results.json")


def token_transitions(model, n_tokens):
    """Per-token mixer inputs -> (M or a, z) from the first block."""
    block = model.stack.blocks[0]
    with torch.no_grad():
        xn = block.norm1(model.emb(torch.arange(n_tokens)))  # (V, D)
        z = torch.sigmoid(block.mixer.linear_z(xn))          # (V, D)
    return block, xn, z


def homomorphism_test(model):
    """Per-block worst-pair Frobenius error of M(g)M(h) vs M(g o h)."""
    block, xn, z = token_transitions(model, 6)
    with torch.no_grad():
        M = block.mixer._transitions(xn.unsqueeze(1))[:, 0]  # (6, n, 2, 2)
        n = M.shape[1]
        # hom error per block: max over all 36 (g, h) pairs
        err = torch.zeros(n)
        for g in range(6):
            for h in range(6):
                gh = COMPOSE[g, h].item()  # g o h
                # sequence h then g applies M(g) @ M(h)
                diff = M[g] @ M[h] - M[gh]
                err = torch.maximum(err, diff.flatten(1).norm(dim=1))
        # faithfulness per block: min pairwise distance among the 6 M(g)
        pair_min = torch.full((n,), float("inf"))
        for g in range(6):
            for h in range(g + 1, 6):
                d = (M[g] - M[h]).flatten(1).norm(dim=1)
                pair_min = torch.minimum(pair_min, d)
        # injection gate per block (mean over the block's 2 channels, 6 tokens)
        z_blocks = z.view(6, n, 2).mean(dim=(0, 2))
        # |det| per block averaged over tokens: exact O(2) elements have 1
        det = torch.linalg.det(M).abs().mean(dim=0)
    best = int(err.argmin())
    order = err.argsort()
    k = max(1, n // 8)  # the automaton needs only a few faithful blocks
    top = order[:k]
    return {
        "n_blocks": n,
        "hom_err_min": round(err.min().item(), 6),
        "hom_err_median": round(err.median().item(), 4),
        "best_block": {
            "index": best,
            "hom_err": round(err[best].item(), 6),
            "faithful_min_pairdist": round(pair_min[best].item(), 4),
            "z_mean": round(z_blocks[best].item(), 6),
            "abs_det_mean": round(det[best].item(), 6),
        },
        "top_blocks_hom_err": [round(err[i].item(), 4) for i in top],
        "top_blocks_z": [round(z_blocks[i].item(), 4) for i in top],
    }


def eigenvalue_test(model):
    """Per-channel (a(0), a(1)) for parity; automaton channels ~ (+1, -1)."""
    block, xn, _ = token_transitions(model, 2)
    with torch.no_grad():
        a, _ = block.mixer._coeffs(xn.unsqueeze(1))
        a = a[:, 0]  # (2, D)
    a0, a1 = a[0], a[1]
    flip = (a1 < -0.99) & (a0 > 0.99)
    near_flip = (a1 < -0.9) & (a0 > 0.9)
    return {
        "channels": a0.numel(),
        "exact_flip_channels": int(flip.sum()),       # (+1, -1) to 1e-2
        "near_flip_channels": int(near_flip.sum()),   # to 1e-1
        "a1_min": round(a1.min().item(), 6),
        "a0_at_a1_min": round(a0[a1.argmin()].item(), 6),
    }


def main():
    os.environ["EXP_CKPT"] = "1"  # winners' headline protocol
    os.environ["EXP_ROUND"] = "verify-mechanism"
    results = {}

    # S3 / rotation-snap: two good seeds and the two failed ones.
    for seed in (0, 1, 2, 6):
        rec, model = run_cell("S3", "rotation-snap", 1, seed, return_model=True)
        results[f"S3_snap_s{seed}"] = {
            "acc": rec["acc"],
            "hom": homomorphism_test(model),
        }

    # parity / signed-tanh: perfect seed and the weakest seed.
    for seed in (0, 2):
        rec, model = run_cell("parity", "signed-tanh", 1, seed, return_model=True)
        results[f"parity_tanh_s{seed}"] = {
            "acc": rec["acc"],
            "eig": eigenvalue_test(model),
        }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
