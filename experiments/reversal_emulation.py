"""Division-reversal error emulation for the chunked reversible backward.

Reproduces, as a committed artifact, the design-phase measurement that led to
rejecting the division-based reversible backward for the rotation-family
mixers (see docs/explanation/givens-mingru.md and the kernels design ledger):
reconstructing hidden states backward through a decayed recurrence by
dividing out the per-step decay amplifies fp32 rounding error like
sigma_min^(-chunklen), where sigma_min is the smallest decay factor crossed
inside a chunk.

Model: the scalar-per-channel core of the rotation-family transition,
h_t = g_t * h_{t-1} + b_t, with per-step decays g_t in [gamma_min, 1).
The rotation part of the real transition is orthogonal, so its inverse is
exact and contributes no amplification; division by g_t is the sole error
driver, which this emulation isolates.

Protocol (fp32 throughout, fp64 only as ground truth):
  - T = 4096 steps, D = 256 channels, chunk length C = 64 (anchors stored
    every C steps, matching the rejected design's checkpoint interval).
  - Forward stores only the chunk anchors; the backward reconstructs
    intra-chunk states from each chunk-end anchor via
    h_{t-1} = (h_t - b_t) / g_t.
  - State error, two metrics: (global) max absolute deviation of the
    reconstructed states normalized by the max state magnitude -- the
    headline; and (strict) per-element max relative deviation, which is
    dominated by near-zero channels and overstates practical damage.
  - Gradient error, same two metrics, on the decay-gate gradient
    dL/dg_t = adj_t * h_{t-1} (L = sum of all states) computed with
    reconstructed vs stored states. This is the quantity the backward
    kernel would actually feed the optimizer.
  - Worst-case bound: eps_fp32 * gamma_min^-(C-1), the constant-decay
    amplification limit, reported alongside for context.

Three decay floors bracket the library's practical range: gamma_min = 0.86
(mild decay), 0.48 (moderate), 0.23 (the default-decay regime that produced
the catastrophic design-phase figure).

Run:  uv run --python 3.12 --with 'torch==2.8.*' python experiments/reversal_emulation.py
Artifacts: experiments/bench/reversal_emulation.json / .md
"""

from __future__ import annotations

import json
import pathlib
import platform

import torch

SEED = 0
T = 4096
D = 256
C = 64
GAMMA_MINS = (0.86, 0.48, 0.23)

_OUT_DIR = pathlib.Path(__file__).resolve().parent / "bench"


def run_one(gamma_min: float) -> dict:
    torch.manual_seed(SEED)
    g64 = torch.rand(T, D, dtype=torch.float64) * (1.0 - gamma_min) + gamma_min
    b64 = torch.randn(T, D, dtype=torch.float64) * 0.1

    g32, b32 = g64.float(), b64.float()

    # fp32 forward (h[0] = 0), storing every state so reconstruction error is
    # measured against exactly what a stored-state backward would have used.
    h32 = torch.zeros(T + 1, D)
    for t in range(T):
        h32[t + 1] = g32[t] * h32[t] + b32[t]

    # Chunked division reversal: from each chunk-end anchor, divide backward.
    rec = h32.clone()
    for start in range(0, T, C):
        end = start + C
        cur = h32[end].clone()
        for t in range(end - 1, start, -1):
            cur = (cur - b32[t]) / g32[t]
            rec[t] = cur

    # Headline metric: max absolute deviation normalized by the max state
    # magnitude (global scale). Per-element relative error is reported as a
    # stricter secondary metric; it is dominated by near-zero channels and
    # overstates practical damage at mild decay.
    state_global = ((rec[1:] - h32[1:]).abs().max() / h32[1:].abs().max()).item()
    denom = h32[1:].abs().clamp_min(torch.finfo(torch.float32).tiny)
    state_rel = ((rec[1:] - h32[1:]).abs() / denom).max().item()

    # Adjoint for L = sum_t h_t: adj_t = 1 + g_{t+1} * adj_{t+1}; the
    # decay-gate gradient is dL/dg_t = adj_t * h_{t-1}.
    adj = torch.zeros(T + 1, D)
    for t in range(T - 1, -1, -1):
        adj[t] = 1.0 + (g32[t + 1] * adj[t + 1] if t + 1 < T else 0.0)
    grad_stored = adj[:T] * h32[:T]
    grad_rec = adj[:T] * rec[:T]
    grad_global = ((grad_rec - grad_stored).abs().max() / grad_stored.abs().max()).item()
    gdenom = grad_stored.abs().clamp_min(torch.finfo(torch.float32).tiny)
    grad_rel = ((grad_rec - grad_stored).abs() / gdenom).max().item()

    bound = torch.finfo(torch.float32).eps * gamma_min ** -(C - 1)
    return {
        "gamma_min": gamma_min,
        "state_err_global": state_global,
        "grad_err_global": grad_global,
        "state_max_rel_err": state_rel,
        "grad_max_rel_err": grad_rel,
        "worst_case_bound_eps_gamma_pow": bound,
    }


def main() -> None:
    rows = [run_one(gm) for gm in GAMMA_MINS]
    payload = {
        "protocol": {
            "T": T,
            "D": D,
            "chunk": C,
            "seed": SEED,
            "decay_sampling": "uniform [gamma_min, 1)",
            "dtype": "float32 (fp64 inputs cast)",
            "loss": "sum of all states",
        },
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "rows": rows,
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "reversal_emulation.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Division-reversal error emulation (chunked reversible backward, C=64)",
        "",
        f"torch {torch.__version__}, CPU, fp32, T={T}, D={D}, chunk C={C}, "
        f"seed {SEED}. Decays sampled uniform in [gamma_min, 1). "
        "See `experiments/reversal_emulation.py` for the protocol.",
        "",
        "| gamma_min | state err (global) | decay-grad err (global) "
        "| state max rel err | grad max rel err "
        "| worst-case bound eps*gamma_min^-(C-1) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['gamma_min']} | {r['state_err_global']:.3e} "
            f"| {r['grad_err_global']:.3e} | {r['state_max_rel_err']:.3e} "
            f"| {r['grad_max_rel_err']:.3e} "
            f"| {r['worst_case_bound_eps_gamma_pow']:.3e} |"
        )
    lines.append("")
    (_OUT_DIR / "reversal_emulation.md").write_text("\n".join(lines))

    for r in rows:
        print(
            f"gamma_min={r['gamma_min']}: state_g={r['state_err_global']:.3e} "
            f"grad_g={r['grad_err_global']:.3e} "
            f"state_rel={r['state_max_rel_err']:.3e} "
            f"grad_rel={r['grad_max_rel_err']:.3e} "
            f"bound={r['worst_case_bound_eps_gamma_pow']:.3e}"
        )


if __name__ == "__main__":
    main()
