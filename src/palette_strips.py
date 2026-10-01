"""The showpiece: one horizontal colour strip per collection.

Each collection's palette is built by pooling the dominant colours of all its
looks (weighted by how much of each look they cover) and re-clustering them.

    python src/palette_strips.py
"""

import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
N_SWATCHES = 6


def hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def collection_palette(looks: pd.DataFrame, k: int = N_SWATCHES) -> list[tuple[np.ndarray, float]]:
    color_cols = sorted(c for c in looks.columns if c.startswith("color_") and not c.endswith("_share"))
    # Long format: one (colour, share) pair per row. Solid looks have fewer
    # than k colours, so their unused slots are empty and get dropped.
    pairs = pd.concat(
        [looks[[c, f"{c}_share"]].set_axis(["hex", "share"], axis=1) for c in color_cols]
    ).dropna()
    colors = np.vstack(pairs["hex"].map(hex_to_rgb).tolist())
    weights = pairs["share"].to_numpy()
    k = min(k, len(colors))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(colors, sample_weight=weights)
    shares = np.bincount(km.labels_, weights=weights, minlength=k)
    palette = [(km.cluster_centers_[i] / 255, shares[i] / shares.sum()) for i in range(k)]
    # Order by hue so strips read like a spectrum; neutrals (greys, near-blacks) go to the end.
    def key(cs):
        h, s, v = colorsys.rgb_to_hsv(*cs[0])
        neutral = s < 0.15 or v < 0.15
        return (neutral, -v if neutral else h)

    return sorted(palette, key=key)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "missoni_dataset.csv")
    df["collection"] = df["year"].astype(str) + " " + df["season"]
    collections = df.sort_values(["year", "season"])["collection"].unique()

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(collections) + 1.4))
    for row, name in enumerate(collections):
        x = 0.0
        for rgb, share in collection_palette(df[df["collection"] == name]):
            ax.add_patch(plt.Rectangle((x, row), share, 0.82, color=rgb, linewidth=0))
            x += share
    ax.set_xlim(0, 1)
    ax.set_ylim(len(collections), -0.4)
    ax.set_yticks(np.arange(len(collections)) + 0.41, collections, fontfamily="monospace")
    ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("MISSONI — YEARS IN COLOR", loc="left", fontsize=15, fontweight="bold", pad=14)
    fig.text(0.125, 0.01, "Swatch width = share of garment area across the collection's looks.", fontsize=8, color="#666")

    out = ROOT / "figures" / "palette_strips.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
