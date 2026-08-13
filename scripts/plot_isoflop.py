"""
Plot the isoFLOP sweep as nanochat's three-panel scaling figure, with the MoE family beside
the dense one.

  left   : val_bpb vs effective params, one parabola per FLOP budget, star at each minimum
  middle : optimal params vs FLOPs, fitted N ~ C^alpha
  right  : optimal tokens vs FLOPs, fitted D ~ C^beta

The middle and right panels are the actual "does it scale" claim -- a clean power law through
the minima says the family behaves predictably over the measured range. The left panel is what
makes the minima trustworthy: a ragged parabola means the fit underneath it is not meaningful.

    python -m scripts.plot_isoflop                       # reads <base_dir>/isoflop.csv
    python -m scripts.plot_isoflop --csv path --out fig.png
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nanochat.common import get_base_dir

# nanochat's own dense fit (dev/LOG.md, Kaplan-style), for reference
NANOCHAT_DENSE = [(1e18, 0.8972), (2e18, 0.8616), (5e18, 0.8293), (1e19, 0.7999)]
FAMILY_STYLE = {  # (label, colour, marker)
    "moe":   ("MoE 8x top-2", "#2a78d6", "o"),
    "dense": ("dense",        "#eb6834", "s"),
}


def parabola_min(x, y):
    """Fit bpb = a*log10(N)^2 + b*log10(N) + c and return the vertex.

    Quadratic in log-space is what nanochat fits, and it is only valid with points either side
    of the minimum -- with 3+ points we fit, with fewer we fall back to the best measured point
    so a thin budget still contributes something rather than producing a fake vertex.
    """
    if len(x) < 3:
        i = int(np.argmin(y))
        return x[i], y[i], None
    lx = np.log10(x)
    a, b, c = np.polyfit(lx, y, 2)
    if a <= 0:                      # not convex -> no interior minimum, fall back
        i = int(np.argmin(y))
        return x[i], y[i], None
    vx = -b / (2 * a)
    return 10 ** vx, a * vx ** 2 + b * vx + c, (a, b, c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--out", default="isoflop.png")
    args = p.parse_args()
    path = args.csv or os.path.join(get_base_dir(), "isoflop.csv")

    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("depth", "n_embd", "n_expert", "top_k",
                                                   "tokens", "iters") else int(float(v)))
                         for k, v in r.items()})
    if not rows:
        raise SystemExit(f"no rows in {path}")

    groups = defaultdict(list)                      # (family, budget) -> rows
    for r in rows:
        groups[("dense" if r["n_expert"] == 1 else "moe", r["budget"])].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    budgets = sorted({r["budget"] for r in rows})
    shades = {b: plt.cm.viridis(i / max(len(budgets) - 1, 1)) for i, b in enumerate(budgets)}
    minima = defaultdict(list)                      # family -> (C, N*, D*, bpb*)

    for (fam, budget), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["eff_params"])
        x = np.array([r["eff_params"] for r in rs])
        y = np.array([r["val_bpb"] for r in rs])
        label, colour, marker = FAMILY_STYLE[fam]
        axes[0].plot(x, y, marker, color=shades[budget], ms=6, ls="none",
                     markeredgecolor="none" if fam == "moe" else "k",
                     markeredgewidth=0.6, alpha=0.9)
        nx, ny, fit = parabola_min(x, y)
        if fit is not None:
            xs = np.logspace(np.log10(x.min()) - .05, np.log10(x.max()) + .05, 120)
            a, b, c = fit
            axes[0].plot(xs, a * np.log10(xs) ** 2 + b * np.log10(xs) + c,
                         "--", color=shades[budget], lw=1.2,
                         alpha=0.9 if fam == "moe" else 0.45)
        axes[0].plot(nx, ny, "*", color=shades[budget], ms=17,
                     markeredgecolor="k", markeredgewidth=0.8, zorder=5)
        # optimal tokens at the vertex: interpolate tokens vs params in log-log
        d_star = 10 ** np.interp(np.log10(nx), np.log10(x),
                                 np.log10([r["tokens"] for r in rs]))
        minima[fam].append((budget, nx, d_star, ny))

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Effective parameters")
    axes[0].set_ylabel("Validation loss (bpb)")
    axes[0].set_title("IsoFLOP curves")
    handles = [plt.Line2D([], [], color=shades[b], marker="o", ls="none", label=f"{b:.0e}")
               for b in budgets]
    handles += [plt.Line2D([], [], color="gray", marker=m, ls="none", label=lbl)
                for lbl, _, m in FAMILY_STYLE.values()]
    axes[0].legend(handles=handles, fontsize=8, title="FLOPs / family")

    for ax, idx, name, ref in ((axes[1], 1, "Optimal parameters", 0.54),
                               (axes[2], 2, "Optimal tokens", 0.49)):
        for fam, pts in minima.items():
            if len(pts) < 2:
                continue
            pts.sort()
            C = np.array([q[0] for q in pts]); V = np.array([q[idx] for q in pts])
            label, colour, marker = FAMILY_STYLE[fam]
            ax.plot(C, V, marker, color=colour, ms=8, ls="none", label=f"{label} (measured)")
            slope, inter = np.polyfit(np.log10(C), np.log10(V), 1)
            cs = np.logspace(np.log10(C.min()) - .15, np.log10(C.max()) + .15, 60)
            ax.plot(cs, 10 ** (inter + slope * np.log10(cs)), "--", color=colour, lw=1.3,
                    label=f"{label}: $\\propto C^{{{slope:.2f}}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("FLOPs"); ax.set_ylabel(name)
        ax.set_title(f"{name} (nanochat dense: $C^{{{ref}}}$)")
        ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    print("\nfitted exponents (nanochat dense reference: N~C^0.54, D~C^0.49)")
    for fam, pts in minima.items():
        if len(pts) < 2:
            print(f"  {fam:6s}: only {len(pts)} budget(s), need >=2 to fit"); continue
        pts.sort()
        C = np.array([q[0] for q in pts])
        a_n = np.polyfit(np.log10(C), np.log10([q[1] for q in pts]), 1)[0]
        a_d = np.polyfit(np.log10(C), np.log10([q[2] for q in pts]), 1)[0]
        print(f"  {fam:6s}: N ~ C^{a_n:.3f}   D ~ C^{a_d:.3f}")
    print("\nminima:")
    for fam, pts in sorted(minima.items()):
        for C, N, D, b in sorted(pts):
            print(f"  {fam:6s} C={C:.1e}  N*={N/1e6:6.1f}M  D*={D/1e9:5.2f}B  bpb={b:.4f}")


if __name__ == "__main__":
    main()
