"""Bootstrap confidence intervals and paired significance tests for the eval results.

Usage: python src/significance.py
Reads results/*.json, writes results/significance.json and results/paired_differences.png.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_BOOT = 10_000
rng = np.random.default_rng(0)

GSM8K = {
    "Baseline": "best_of_n_baseline.json",
    "ORM": "best_of_n_models_orm-trained_checkpoint-2400.json",
    "PRM-human": "best_of_n_models_prm-human-trained_checkpoint-2400.json",
    "PRM-auto": "best_of_n_models_prm-auto-gsm8k-trained.json",
}
MATH = {
    "Baseline": "math500_baseline.json",
    "ORM": "math500_models_orm-math-trained_checkpoint-3100.json",
    "PRM-human": "math500_models_prm-human-math-trained.json",
    "PRM-auto": "math500_models_prm-auto-math-trained.json",
}


def load(files, key, qkey):
    out, order = {}, None
    for name, f in files.items():
        res = json.loads((RESULTS / f).read_text())["results"]
        qs = [r[qkey] for r in res]
        if order is None:
            order = qs
        assert qs == order, f"{f}: problems not in the same order as baseline"
        out[name] = np.array([bool(r[key]) for r in res], dtype=float)
    return out, res


def ci(x):
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(a, b):
    """Paired bootstrap CI of mean(a - b) plus exact McNemar p-value."""
    d = a - b
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(1)
    n01 = int(((a == 1) & (b == 0)).sum())  # a right, b wrong
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"diff_pp": 100 * d.mean(), "ci_pp": [100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)],
            "a_only": n01, "b_only": n10, "mcnemar_p": p}


def analyse(files, key, qkey, label):
    data, _ = load(files, key, qkey)
    summary = {}
    for name, x in data.items():
        mu, lo, hi = ci(x)
        summary[name] = {"acc": 100 * mu, "ci": [100 * lo, 100 * hi]}
    comps = {}
    for a in ["ORM", "PRM-human", "PRM-auto"]:
        comps[f"{a} vs Baseline"] = paired(data[a], data["Baseline"])
    comps["PRM-human vs ORM"] = paired(data["PRM-human"], data["ORM"])
    comps["PRM-human vs PRM-auto"] = paired(data["PRM-human"], data["PRM-auto"])
    return summary, comps, data


def forest(d):
    rows = ["ORM vs Baseline", "PRM-human vs Baseline", "PRM-auto vs Baseline", "PRM-human vs PRM-auto"]
    labels = ["ORM − Baseline", "PRM-human − Baseline", "PRM-auto − Baseline", "PRM-human − PRM-auto"]
    panels = [("GSM8K  (best-of-8, n=200)", d["gsm8k_best_of_8_majority"]["paired"]),
              ("MATH-500 subset  (greedy, n=200)", d["math500_greedy"]["paired"])]
    ink, muted, grid, accent = "#1f2328", "#6e7781", "#d0d7de", "#2f6fdb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200, sharey=True)
    for ax, (title, comp) in zip(axes, panels):
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            c = comp[r]; lo, hi = c["ci_pp"]
            sig = c["mcnemar_p"] < 0.05
            col = accent if sig else muted
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(c["diff_pp"], y, "o", ms=8, color=col, mec="white", mew=2, zorder=3)
            ax.text(1.03, y, f"p = {c['mcnemar_p']:.3f}" if c["mcnemar_p"] < 0.1 else f"p = {c['mcnemar_p']:.2f}",
                    va="center", ha="left", fontsize=8.5, transform=ax.get_yaxis_transform(), color=ink if sig else muted)
        ax.axvline(0, color=ink, lw=1)
        ax.set_xlim(-5, 11.5); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("Accuracy difference (percentage points)", color=muted, fontsize=9)
        ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(labels[::-1], color=ink)
    fig.text(0.01, 0.01, "Dot = difference on the same 200 problems. Line = 95% paired bootstrap CI. Blue = significant at p < 0.05 (exact McNemar test).",
             fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=5)
    fig.savefig(RESULTS / "paired_differences.png")


if __name__ == "__main__":
    out = {}
    g_sum, g_cmp, _ = analyse(GSM8K, "majority_correct", "question", "gsm8k")
    m_sum, m_cmp, m_data = analyse(MATH, "correct", "problem", "math")
    math_res = json.loads((RESULTS / MATH["Baseline"]).read_text())["results"]
    levels = np.array([r["level"] for r in math_res])
    per_level = {str(l): {n: {"correct": int(x[levels == l].sum()), "n": int((levels == l).sum())}
                          for n, x in m_data.items()} for l in sorted(set(levels.tolist()))}
    out = {"gsm8k_best_of_8_majority": {"accuracy": g_sum, "paired": g_cmp},
           "math500_greedy": {"accuracy": m_sum, "paired": m_cmp, "per_level_counts": per_level},
           "method": f"Percentile bootstrap, {N_BOOT} resamples over problems, seed 0; exact McNemar test."}
    (RESULTS / "significance.json").write_text(json.dumps(out, indent=2))
    forest(out)
    for label, s, c in [("GSM8K", g_sum, g_cmp), ("MATH", m_sum, m_cmp)]:
        print(f"\n== {label} ==")
        for n, v in s.items():
            print(f"  {n:10s} {v['acc']:.1f}% [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}]")
        for n, v in c.items():
            print(f"  {n:24s} {v['diff_pp']:+.1f}pp [{v['ci_pp'][0]:+.1f}, {v['ci_pp'][1]:+.1f}]  "
                  f"wins {v['a_only']}/{v['b_only']}  McNemar p={v['mcnemar_p']:.3f}")
    print("\nMATH per-level (correct/n):")
    for l, d in per_level.items():
        print(f"  L{l}: " + "  ".join(f"{n} {v['correct']}/{v['n']}" for n, v in d.items()))
