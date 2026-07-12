"""Expressivity probes: GRU vs minGRU vs SignedMinGRU.

Tasks (seq2seq tagging, dense supervision as in Merrill et al. 2024):
  parity : running XOR over {0,1}. In TC0; the natural one-scan solution
           needs a transition eigenvalue at -1, so positive-diagonal
           minGRU lacks it while SignedMinGRU and GRU have it.
  S3     : running product in the symmetric group S3 (smallest
           non-abelian group). Order-sensitive; commutative (diagonal)
           scans of ANY sign should fail at one layer, while a GRU
           encodes the 6-state automaton directly.

Train at T=64; evaluate at T=64 (in-dist) and T=256 (length gen).

Usage: python probes.py TASK MODEL [N_LAYERS]
       TASK in {parity, S3}; MODEL in {GRU, minGRU, minGRU-signed};
       N_LAYERS defaults to 1 (applies to all models).
       MAX_STEPS env var overrides the training budget (default 1600).
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from min_gru import MinGRUStack

D_MODEL = 64
T_TRAIN, T_GEN = 64, 256
BATCH = 128
MAX_STEPS = int(os.environ.get("MAX_STEPS", 1600))
EVAL_EVERY = 100
LR = 3e-3

# ---------------------------------------------------------------- tasks
def make_parity(batch, T, gen):
    x = torch.randint(0, 2, (batch, T), generator=gen)
    y = x.cumsum(dim=1) % 2
    return x, y

# S3 as permutations of (0,1,2); element 0 is the identity.
S3 = torch.tensor(
    [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [2, 0, 1]]
)

def _compose_table():
    idx = {tuple(p.tolist()): k for k, p in enumerate(S3)}
    table = torch.zeros(6, 6, dtype=torch.long)
    for i in range(6):
        for j in range(6):
            comp = S3[i][S3[j]]  # p_i o p_j
            table[i, j] = idx[tuple(comp.tolist())]
    return table

COMPOSE = _compose_table()

def make_s3(batch, T, gen):
    x = torch.randint(0, 6, (batch, T), generator=gen)
    y = torch.zeros_like(x)
    state = torch.zeros(batch, dtype=torch.long)  # identity
    for t in range(T):
        state = COMPOSE[x[:, t], state]  # g_t o r_{t-1}
        y[:, t] = state
    return x, y

TASKS = {"parity": (make_parity, 2, 2), "S3": (make_s3, 6, 6)}

# ---------------------------------------------------------------- models
class GRUTagger(nn.Module):
    def __init__(self, vocab, n_cls, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.rnn = nn.GRU(D_MODEL, D_MODEL, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h)


class MinGRUTagger(nn.Module):
    def __init__(self, vocab, n_cls, signed, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.stack = MinGRUStack(D_MODEL, D_MODEL, n_layers=n_layers, signed=signed)
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        return self.head(self.stack(self.emb(x)))


def build(name, vocab, n_cls, n_layers=1):
    if name == "GRU":
        return GRUTagger(vocab, n_cls, n_layers)
    return MinGRUTagger(vocab, n_cls, signed=(name == "minGRU-signed"), n_layers=n_layers)


@torch.no_grad()
def accuracy(model, make, T, seed, n_batches=4):
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        x, y = make(BATCH, T, gen)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def run_one(task, name, n_layers=1, max_steps=MAX_STEPS):
    make, vocab, n_cls = TASKS[task]
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    model = build(name, vocab, n_cls, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0, steps_used = time.time(), max_steps
    for step in range(1, max_steps + 1):
        x, y = make(BATCH, T_TRAIN, gen)
        loss = F.cross_entropy(model(x).reshape(-1, n_cls), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            if accuracy(model, make, T_TRAIN, seed=2, n_batches=2) >= 0.999:
                steps_used = step
                break
    acc_in = accuracy(model, make, T_TRAIN, seed=3)
    acc_gen = accuracy(model, make, T_GEN, seed=4)
    print(
        f"{task:>7} | {name:<14} | L={n_layers} | acc@{T_TRAIN}: {acc_in:.3f} | "
        f"acc@{T_GEN}: {acc_gen:.3f} | steps: {steps_used:>4} | "
        f"{time.time() - t0:5.1f}s",
        flush=True,
    )


GRID = [
    # (task, model, n_layers, max_steps)
    ("parity", "GRU", 1, 1500),
    ("parity", "minGRU", 1, 1500),
    ("parity", "minGRU-signed", 1, 1500),
    ("S3", "GRU", 1, 1500),
    ("S3", "minGRU", 1, 1500),
    ("S3", "minGRU-signed", 1, 1500),
    ("parity", "minGRU", 4, 1600),
    ("parity", "minGRU-signed", 4, 1600),
    ("S3", "minGRU", 4, 1600),
    ("S3", "minGRU-signed", 4, 1600),
]


def run_grid():
    # MAX_STEPS env var, when set, overrides the per-entry grid budgets
    # (docstring contract); otherwise each entry uses its own budget.
    env_override = "MAX_STEPS" in os.environ
    for task, name, n_layers, max_steps in GRID:
        run_one(task, name, n_layers, MAX_STEPS if env_override else max_steps)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_grid()
    else:
        run_one(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
