"""
IsoFLOP sweep: reproduce nanochat's scaling-law figure with an MoE family beside the dense one.

For each FLOP budget C we train several model sizes to exactly C FLOPs (via base_train's
--target-flops) and record final val_bpb. Plotting bpb against effective parameters gives one
parabola per budget; its minimum is the compute-optimal model at that budget. Fitting the minima
across budgets gives N ~ C^alpha and D ~ C^beta.

Why this is worth running: a single run per budget -- which is all we have today -- cannot locate
a minimum, so "we are compute-optimal" and "we beat the dense frontier" both rest on nanochat's
published dense law holding outside its fitted range. A sweep replaces both inferences with
measurements, in our own harness, at our own vocab.

Runs inside ONE container and loops, so the ~2 min of container start + torch.compile warmup is
paid once rather than per run. Resumable: rows already in the CSV are skipped, so a killed sweep
picks up where it left off.

    python -m scripts.isoflop_sweep --budgets 3e16 1e17 3e17
    python -m scripts.isoflop_sweep --budgets 3e16 --families moe    # cheap smoke test
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

from nanochat.common import get_base_dir
from scripts.size_moe import size

# depth ladders per budget, chosen to bracket the predicted optimum on both sides -- a parabola
# needs points either side of its minimum or the fit is meaningless.
DEPTH_LADDER = {
    3e16: [4, 6, 8, 10, 12],
    1e17: [6, 8, 10, 12, 16],
    3e17: [8, 10, 12, 16, 20],
    1e18: [10, 12, 16, 20, 24],
}
ASPECT, HEAD_DIM, VOCAB = 32, 64, 8192


def model_dim(depth):
    return ((depth * ASPECT + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM


def effective_params(depth, n_expert, top_k):
    """x-axis of the isoFLOP plot.

    Dense: Kaplan-style params (transformer matrices + lm_head, no embeddings), matching
    nanochat's own convention. MoE: sqrt(active * total) -- the geometric mean, which is the
    heuristic our two existing runs support (it predicted 21.8M against a 23.6M dense-optimal
    at v1's compute, 7% off, and v1 landed on the dense frontier as that implies).
    """
    s = size(model_dim(depth), depth, vocab_size=VOCAB, head_dim=HEAD_DIM,
             n_expert=n_expert, top_k=top_k)
    if n_expert == 1:
        return s["nonembed_total"]
    return math.sqrt(s["nonembed_active"] * s["nonembed_total"])


def run_one(budget, depth, n_expert, top_k, extra):
    tag = f"iso-c{budget:.0e}-d{depth}-e{n_expert}".replace("+", "")
    cmd = [
        sys.executable, "-u", "-m", "scripts.base_train",
        f"--target-flops={budget}", "--target-param-data-ratio=-1",
        f"--depth={depth}", f"--aspect-ratio={ASPECT}", f"--head-dim={HEAD_DIM}",
        "--window-pattern=L", f"--n-expert={n_expert}", f"--top-k={top_k}",
        "--moe-dispatch=grouped" if n_expert > 1 else "--moe-dispatch=loop",
        f"--model-tag={tag}",
        # one val at the end is all the sweep needs; CORE and sampling are pure cost here
        "--eval-every=100000", "--eval-tokens=10485760",
        "--core-metric-every=-1", "--sample-every=-1", "--expert-load-every=-1",
        "--device-batch-size=16",
    ] + extra
    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out[-3000:], flush=True)
        raise RuntimeError(f"{tag} failed with code {proc.returncode}")

    bpb = re.findall(r"Minimum validation bpb:\s*([0-9.]+)", out)
    tokens = re.findall(r"Total number of training tokens:\s*([\d,]+)", out)
    iters = re.findall(r"Calculated number of iterations from target FLOPs:\s*([\d,]+)", out)
    if not bpb:
        print(out[-3000:], flush=True)
        raise RuntimeError(f"{tag}: could not parse val_bpb")
    row = dict(
        budget=budget, depth=depth, n_embd=model_dim(depth), n_expert=n_expert, top_k=top_k,
        eff_params=effective_params(depth, n_expert, top_k),
        tokens=int(tokens[-1].replace(",", "")) if tokens else -1,
        iters=int(iters[-1].replace(",", "")) if iters else -1,
        val_bpb=float(bpb[-1]), wall_s=round(time.time() - t0, 1),
    )
    print(f"-> bpb {row['val_bpb']:.4f}  N_eff {row['eff_params'] / 1e6:.1f}M  "
          f"{row['tokens'] / 1e6:.0f}M tokens  {row['wall_s'] / 60:.1f} min", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description="IsoFLOP sweep for MoE vs dense")
    p.add_argument("--budgets", type=float, nargs="+", default=[3e16, 1e17, 3e17])
    p.add_argument("--families", nargs="+", default=["moe", "dense"], choices=["moe", "dense"])
    p.add_argument("--n-expert", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--out", type=str, default=None, help="CSV path (default: <base_dir>/isoflop.csv)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra flags forwarded verbatim to base_train")
    args = p.parse_args()

    out_path = args.out or os.path.join(get_base_dir(), "isoflop.csv")
    fields = ["budget", "depth", "n_embd", "n_expert", "top_k", "eff_params",
              "tokens", "iters", "val_bpb", "wall_s"]
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for r in csv.DictReader(f):
                done.add((float(r["budget"]), int(r["depth"]), int(r["n_expert"])))
        print(f"resuming: {len(done)} rows already in {out_path}")
    else:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    plan = []
    for budget in args.budgets:
        ladder = DEPTH_LADDER.get(budget) or DEPTH_LADDER[min(DEPTH_LADDER, key=lambda b: abs(b - budget))]
        for fam in args.families:
            ne, tk = (args.n_expert, args.top_k) if fam == "moe" else (1, 1)
            for depth in ladder:
                if (budget, depth, ne) not in done:
                    plan.append((budget, depth, ne, tk))
    print(f"{len(plan)} runs to do")

    for i, (budget, depth, ne, tk) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}]", flush=True)
        row = run_one(budget, depth, ne, tk, args.extra)
        with open(out_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
    print(f"\ndone -> {out_path}")


if __name__ == "__main__":
    main()
