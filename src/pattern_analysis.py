"""Chapter 02, part 2: pattern types.

Pattern labels in looks.csv mark each look's most prominent pattern. They are
an AI-assisted first pass (see data/README.md) and should be reviewed.

1. The pattern mix of each collection.
2. Can a classifier learn pattern type from garment-only CLIP embeddings?
   Evaluated leave-one-collection-out: train on 8 collections, predict the 9th.

    python src/pattern_analysis.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from chapter_charts import GRID, INK, INK_SECONDARY, SURFACE, save, titled

ROOT = Path(__file__).resolve().parent.parent
# Fixed order = fixed colour: categorical slots 1–6 of the dataviz reference palette.
PATTERNS = {
    "zigzag": "#2a78d6", "stripe": "#eb6834", "space-dye": "#1baf7a",
    "geometric": "#eda100", "print": "#e87ba4", "plain": "#008300",
}
KNIT = {"zigzag", "stripe", "space-dye"}
N_PERMUTATIONS = 100


def load() -> tuple[pd.DataFrame, np.ndarray]:
    looks = pd.read_csv(ROOT / "data" / "looks.csv")
    looks["collection"] = looks["year"].astype(str) + " " + looks["season"]
    emb = pd.read_csv(ROOT / "data" / "embeddings_clip.csv").set_index("look_id").loc[looks["look_id"]]
    return looks, emb.to_numpy()


def pattern_mix(looks: pd.DataFrame) -> None:
    names = list(looks.sort_values(["year", "season"])["collection"].unique())
    mix = pd.crosstab(looks["collection"], looks["pattern"]).reindex(index=names, columns=list(PATTERNS), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.11, right=0.97)
    ax.set_facecolor(SURFACE)
    for row, name in enumerate(names):
        x = 0
        for pattern, color in PATTERNS.items():
            n = mix.loc[name, pattern]
            if n:
                ax.barh(row, n - 0.12, left=x, height=0.66, color=color, linewidth=0)
                if n >= 3:
                    ax.text(x + (n - 0.12) / 2, row, str(n), ha="center", va="center", fontsize=9, color="white",
                            fontweight="bold")
            x += n
    handles = [Patch(color=c, label=p) for p, c in PATTERNS.items()]
    ax.legend(handles=handles, frameon=False, ncol=6, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=9,
              labelcolor=INK_SECONDARY, handlelength=1.2, columnspacing=1.2)
    ax.set_yticks(range(len(names)), names, fontfamily="monospace")
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(0, 15)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Looks (of 15 sampled)")
    ax.tick_params(length=0, colors=INK_SECONDARY)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    titled(fig, "02b — The zigzag returns in FW 2023",
           "Most prominent pattern of each sampled look. 2008 and 2014 are print collections; FW 2016 is the stripe show.",
           "Labels: AI-assisted first pass from garment close-ups (data/README.md); to be reviewed.")
    save(fig, "ch02b_pattern_mix.png")


def classify(looks: pd.DataFrame, X: np.ndarray) -> dict:
    y = looks["pattern"].to_numpy()
    groups = looks["collection"].to_numpy()
    model = LogisticRegression(C=10, max_iter=5000, class_weight="balanced")
    pred = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=groups)
    bal = balanced_accuracy_score(y, pred)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERMUTATIONS):
        yp = rng.permutation(y)
        null.append(balanced_accuracy_score(yp, cross_val_predict(model, X, yp, cv=LeaveOneGroupOut(), groups=groups)))
    coarse = lambda v: np.array(["knit pattern" if p in KNIT else p for p in v])  # noqa: E731
    return {
        "accuracy": round(accuracy_score(y, pred), 3),
        "majority_baseline": round(pd.Series(y).value_counts(normalize=True).max(), 3),
        "balanced_accuracy": round(bal, 3),
        "balanced_chance": round(float(np.mean(null)), 3),
        "p_value": round(float((np.sum(np.array(null) >= bal) + 1) / (len(null) + 1)), 4),
        "coarse_accuracy": round(accuracy_score(coarse(y), coarse(pred)), 3),
        "coarse_baseline": round(pd.Series(coarse(y)).value_counts(normalize=True).max(), 3),
        "recall": pd.Series(pred == y, index=y).groupby(level=0).mean().round(3).to_dict(),
        "confusion": confusion_matrix(y, pred, labels=list(PATTERNS)).tolist(),
    }


def confusion_chart(res: dict) -> None:
    labels = list(PATTERNS)
    cm = np.array(res["confusion"])
    rates = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    fig.subplots_adjust(top=0.8, bottom=0.14, left=0.17, right=0.97)
    ax.set_facecolor(SURFACE)
    cmap = plt.get_cmap("Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            r = rates[i, j]
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.96, color=cmap(0.08 + 0.85 * r) if cm[i, j] else GRID,
                                       linewidth=0))
            if cm[i, j]:
                ax.text(j + 0.48, i + 0.48, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if r > 0.45 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xlim(0, len(labels))
    ax.set_ylim(len(labels), 0)
    ax.set_xticks(np.arange(len(labels)) + 0.48, labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)) + 0.48, labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Labelled")
    ax.tick_params(length=0, colors=INK_SECONDARY)
    for side in ax.spines.values():
        side.set_visible(False)
    titled(fig, "Prints are easy; zigzag vs stripe is where it fails",
           f"Pattern classifier on unseen collections: {res['accuracy']:.0%} correct "
           f"(always guessing 'print': {res['majority_baseline']:.0%}).",
           f"Logistic regression on garment-only CLIP embeddings, leave-one-collection-out. Balanced accuracy "
           f"{res['balanced_accuracy']:.0%} vs {res['balanced_chance']:.0%} chance (p = {res['p_value']}).")
    save(fig, "ch02c_pattern_classifier.png")


def main() -> None:
    looks, X = load()
    pattern_mix(looks)
    res = classify(looks, X)
    confusion_chart(res)
    (ROOT / "data" / "pattern_results.json").write_text(json.dumps(res, indent=2))
    print({k: v for k, v in res.items() if k not in ("confusion", "recall")})


if __name__ == "__main__":
    main()
