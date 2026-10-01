"""Chapters 04–05: how alike are collections, and can the looks reveal their own eras?

Uses the per-look embeddings from embeddings.py.

    python src/era_analysis.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from chapter_charts import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE, frame, save, titled

ROOT = Path(__file__).resolve().parent.parent
# Ordinal blue ramp from the dataviz reference palette (light -> dark = older -> newer).
YEAR_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
EDGE_DIR_WEIGHT = 0.7  # colour carries most of the signal; edge directions add a little
N_PERMUTATIONS = 200


def load(kind: str = "handcrafted") -> tuple[np.ndarray, pd.DataFrame]:
    emb = pd.read_csv(ROOT / "data" / f"embeddings_{kind}.csv")
    looks = pd.read_csv(ROOT / "data" / "looks.csv").set_index("look_id").loc[emb["look_id"]].reset_index()
    looks["collection"] = looks["year"].astype(str) + " " + looks["season"]
    if kind == "handcrafted":
        X = np.hstack([emb.filter(like="color_").to_numpy(), EDGE_DIR_WEIGHT * emb.filter(like="edge_dir").to_numpy()])
    else:
        X = emb.drop(columns="look_id").to_numpy()
    return X, looks


def collection_order(looks: pd.DataFrame) -> list[str]:
    return list(looks.sort_values(["year", "season"])["collection"].unique())


def recognisability(X: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """Leave-one-out 5-nearest-neighbour accuracy at naming each look's collection,
    with a permutation test: how often do shuffled labels do as well?"""
    knn = KNeighborsClassifier(5)
    pred = cross_val_predict(knn, X, labels, cv=LeaveOneOut())
    acc = float((pred == labels).mean())
    rng = np.random.default_rng(seed)
    null = np.array([
        (cross_val_predict(knn, X, p, cv=LeaveOneOut()) == p).mean()
        for p in (rng.permutation(labels) for _ in range(N_PERMUTATIONS))
    ])
    per = pd.Series(pred == labels, index=labels).groupby(level=0).mean()
    return {
        "accuracy": round(acc, 3), "chance": round(float(null.mean()), 3),
        "p_value": round(float((np.sum(null >= acc) + 1) / (len(null) + 1)), 4),
        "per_collection": per.round(3).to_dict(),
    }


def distance_matrix(X: np.ndarray, labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Energy-style distance between collections: how much further apart their looks are
    than looks within each collection. 0 = indistinguishable; larger = more different."""
    D = cdist(X, X)
    mean = {(a, b): D[np.ix_(labels == a, labels == b)].mean() for a in names for b in names}
    return pd.DataFrame(
        [[mean[a, b] - (mean[a, a] + mean[b, b]) / 2 for b in names] for a in names], index=names, columns=names
    )


def ch04_similarity(dist: pd.DataFrame, suffix: str) -> None:
    names = list(dist.index)
    n = len(names)
    fig, ax = plt.subplots(figsize=(8.6, 8), facecolor=SURFACE)
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.14, right=0.97)
    ax.set_facecolor(SURFACE)
    vals = dist.to_numpy()
    off = vals[~np.eye(n, dtype=bool)]
    lo, hi = off.min(), off.max()
    cmap = plt.get_cmap("Blues_r")
    for i in range(n):
        for j in range(i):
            v = vals[i, j]
            t = (v - lo) / (hi - lo)  # 0 = most similar pair
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.96, color=cmap(0.15 + 0.75 * t), linewidth=0))
            ax.text(j + 0.48, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if t < 0.45 else INK)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(n, 1)
    ax.set_xticks(np.arange(n - 1) + 0.48, names[:-1], rotation=45, ha="right", fontfamily="monospace")
    ax.set_yticks(np.arange(1, n) + 0.48, names[1:], fontfamily="monospace")
    ax.tick_params(length=0, colors=INK_SECONDARY)
    for side in ax.spines.values():
        side.set_visible(False)
    pairs = [(vals[i, j], names[i], names[j]) for i in range(n) for j in range(i)]
    (d1, a1, b1), (d2, a2, b2) = min(pairs), max(pairs)
    titled(fig, "04 — Which collections look most alike?",
           f"Closest: {b1} & {a1}. Furthest apart: {b2} & {a2}. Darker = more similar.",
           "Distance between collections minus their internal spread, on color + edge-direction fingerprints.")
    save(fig, f"ch04_similarity{suffix}.png")


def ch05_map(X: np.ndarray, looks: pd.DataFrame, rec: dict, suffix: str) -> None:
    pca = PCA(2, random_state=0).fit(X)
    Z = pca.transform(X)
    names = collection_order(looks)
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.05, right=0.8)
    frame(ax, "Visual axis 2")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Visual axis 1")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    centres = []
    for name, color in zip(names, YEAR_RAMP):
        m = (looks["collection"] == name).to_numpy()
        ax.scatter(Z[m, 0], Z[m, 1], s=26, color=color, alpha=0.55, linewidths=0)
        centres.append(Z[m].mean(axis=0))
    centres = np.array(centres)
    # The path through the collection centres in chronological order.
    ax.plot(centres[:, 0], centres[:, 1], color=INK_MUTED, linewidth=1.2, zorder=2)
    for (cx, cy), name, color in zip(centres, names, YEAR_RAMP):
        ax.scatter(cx, cy, s=120, color=color, edgecolors=INK, linewidths=1, zorder=3, label=name)
        offset = {"2011 FW": (8, -14), "2016 FW": (8, -14)}.get(name, (8, 5))
        ax.annotate(name, (cx, cy), xytext=offset, textcoords="offset points", fontsize=9,
                    fontfamily="monospace", color=INK, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.8))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, labelcolor=INK_SECONDARY,
              title="Collection (centre)", title_fontsize=9, prop={"family": "monospace", "size": 9})
    acc, chance = rec["accuracy"], rec["chance"]
    titled(fig, "05 — Each show has a signature, but there's no timeline",
           f"Collection centres jump around rather than drifting in one direction (line = chronological order).\n"
           f"Yet a nearest-neighbour test names a look's collection {acc:.0%} of the time vs {chance:.0%} by chance.",
           f"PCA of per-look fingerprints (colour + edge direction); the 2 axes capture {pca.explained_variance_ratio_.sum():.0%} "
           "of the variation. Small dots = looks, ringed dots = collection centres.")
    save(fig, f"ch05_era_map{suffix}.png")


def ch05_distinctiveness(rec: dict, names: list[str], suffix: str) -> None:
    per = pd.Series(rec["per_collection"]).reindex(names)
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.11, right=0.95)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(names))
    ax.barh(y, per.to_numpy(), height=0.6, color="#2a78d6")
    ax.axvline(rec["chance"], color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(rec["chance"], -0.7, f" chance ({rec['chance']:.0%})", color=INK_MUTED, fontsize=8.5, va="bottom")
    for yi, v in zip(y, per):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y, names, fontfamily="monospace")
    ax.set_ylim(len(names) - 0.5, -0.9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_SECONDARY)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    top = per.idxmax()
    titled(fig, f"How recognisable is each collection? {top} most of all",
           "Share of a collection's looks whose 5 nearest neighbours mostly come from the same show.",
           f"Leave-one-out 5-NN on per-look fingerprints. Overall {rec['accuracy']:.0%} vs {rec['chance']:.0%} chance "
           f"(permutation test, p = {rec['p_value']}).")
    save(fig, f"ch05_distinctiveness{suffix}.png")


def run(kind: str = "handcrafted") -> dict:
    suffix = "" if kind == "handcrafted" else f"_{kind}"
    X, looks = load(kind)
    labels = looks["collection"].to_numpy()
    names = collection_order(looks)
    rec = recognisability(X, labels)
    dist = distance_matrix(X, labels, names)
    pca = PCA(2, random_state=0).fit(X)
    rec["pca_year_corr"] = [round(float(np.corrcoef(c, looks["year"])[0, 1]), 3) for c in pca.transform(X).T]
    ch04_similarity(dist, suffix)
    ch05_map(X, looks, rec, suffix)
    ch05_distinctiveness(rec, names, suffix)
    out = ROOT / "data" / f"era_results_{kind}.json"
    out.write_text(json.dumps({**rec, "distance": dist.round(3).to_dict()}, indent=2))
    print(f"-> {out.relative_to(ROOT)}: {rec['accuracy']:.0%} vs {rec['chance']:.0%} chance, p={rec['p_value']}")
    return rec


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "handcrafted")
