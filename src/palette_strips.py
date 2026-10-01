"""The showpiece: one horizontal colour strip per collection.

Two charts:

- palette_strips.png — per collection, a strip of its main colours (width =
  share of garment area) plus its most vivid accent colours on the right.
- look_palettes.png — every sampled look as its own small colour bar, so the
  variety inside a collection stays visible.

Colours are grouped in CIELAB, where distance matches perceived difference,
and each swatch is a real colour taken from a look (the medoid of its group)
rather than an average — averaging bright multicolour knits produces mud.

    python src/palette_strips.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from visual_features import to_lab

ROOT = Path(__file__).resolve().parent.parent
N_SWATCHES = 6
N_ACCENTS = 4
ACCENT_MIN_CHROMA = 30  # CIELAB chroma; below this a colour reads as a neutral
ACCENT_MIN_DISTANCE = 25  # CIELAB ΔE between accents, so they aren't near-duplicates

INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def look_colors(looks: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (look, colour) with its share of that look's garment."""
    color_cols = sorted(c for c in looks.columns if c.startswith("color_") and not c.endswith("_share"))
    pairs = pd.concat(
        [looks[["look_id", c, f"{c}_share"]].set_axis(["look_id", "hex", "share"], axis=1) for c in color_cols]
    ).dropna()
    rgb = np.vstack(pairs["hex"].map(hex_to_rgb).tolist())
    lab = to_lab(rgb)
    pairs[["L", "a", "b"]] = lab
    pairs["chroma"] = np.hypot(lab[:, 1], lab[:, 2])
    pairs["hue"] = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    return pairs.reset_index(drop=True)


def main_palette(colors: pd.DataFrame, k: int = N_SWATCHES) -> pd.DataFrame:
    """k colour groups by garment area; each represented by its medoid (a real look colour)."""
    lab = colors[["L", "a", "b"]].to_numpy()
    k = min(k, len(colors))
    km = KMeans(n_clusters=k, n_init=8, random_state=0).fit(lab, sample_weight=colors["share"])
    out = []
    for c in range(k):
        members = colors[km.labels_ == c]
        centre = km.cluster_centers_[c]
        medoid = members.iloc[np.linalg.norm(members[["L", "a", "b"]].to_numpy() - centre, axis=1).argmin()]
        out.append({**medoid.to_dict(), "weight": members["share"].sum()})
    pal = pd.DataFrame(out)
    pal["weight"] /= pal["weight"].sum()
    # Chromatic colours ordered by hue (a spectrum), then neutrals light -> dark.
    pal["neutral"] = pal["chroma"] < 12
    return pal.sort_values(["neutral", "hue", "L"], ascending=[True, True, False])


def accents(colors: pd.DataFrame, n: int = N_ACCENTS) -> pd.DataFrame:
    """The most vivid colours in the collection, each clearly different from the others."""
    picked = []
    for _, c in colors[colors["chroma"] >= ACCENT_MIN_CHROMA].sort_values("chroma", ascending=False).iterrows():
        lab = c[["L", "a", "b"]].to_numpy(float)
        if all(np.linalg.norm(lab - p[["L", "a", "b"]].to_numpy(float)) >= ACCENT_MIN_DISTANCE for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    return pd.DataFrame(picked).sort_values("hue") if picked else pd.DataFrame(columns=colors.columns)


def style(ax) -> None:
    ax.set_xticks([])
    ax.tick_params(length=0, colors=INK)
    for side in ax.spines.values():
        side.set_visible(False)


def collections(df: pd.DataFrame) -> list[str]:
    df["collection"] = df["year"].astype(str) + " " + df["season"]
    return list(df.sort_values(["year", "season"])["collection"].unique())


def palette_strips(df: pd.DataFrame, colors: pd.DataFrame) -> None:
    names = collections(df)
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(names) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    gap, acc_x0, acc_w = 0.003, 1.06, 0.065  # strip spans 0–1; accent squares to its right
    for row, name in enumerate(names):
        ids = df.loc[df["collection"] == name, "look_id"]
        c = colors[colors["look_id"].isin(ids)]
        x = 0.0
        for _, sw in main_palette(c).iterrows():
            ax.add_patch(plt.Rectangle((x, row), sw["weight"] - gap, 0.8, color=sw["hex"], linewidth=0))
            x += sw["weight"]
        for j, (_, sw) in enumerate(accents(c).iterrows()):
            ax.add_patch(plt.Rectangle((acc_x0 + j * acc_w, row), acc_w - 0.008, 0.8, color=sw["hex"], linewidth=0))

    ax.set_xlim(0, acc_x0 + N_ACCENTS * acc_w)
    ax.set_ylim(len(names), -0.9)
    ax.set_yticks(np.arange(len(names)) + 0.4, names, fontfamily="monospace", fontsize=11)
    style(ax)
    ax.text(0, -0.3, "PALETTE  ·  width = share of garment area", fontsize=8.5, color=INK_MUTED)
    ax.text(acc_x0, -0.3, "ACCENTS  ·  most vivid", fontsize=8.5, color=INK_MUTED)
    ax.set_title("MISSONI — YEARS IN COLOR", loc="left", fontsize=16, fontweight="bold", color=INK, pad=18)
    fig.text(
        0.125, 0.015,
        "15 runway looks per collection, garments isolated by segmentation. Each swatch is a real color "
        "from the looks (CIELAB medoid), not an average.",
        fontsize=8, color=INK_MUTED,
    )
    save(fig, "palette_strips.png")


def look_palettes(df: pd.DataFrame, colors: pd.DataFrame) -> None:
    """Grid: one row per collection, one small vertical colour bar per look."""
    names = collections(df)
    per_row = df.groupby("collection").size().max()
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(names) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for row, name in enumerate(names):
        looks = df[df["collection"] == name].sort_values("look_number")
        for col, look_id in enumerate(looks["look_id"]):
            y = row
            sw = colors[colors["look_id"] == look_id].sort_values("share", ascending=False)
            for _, c in sw.iterrows():
                h = 0.85 * c["share"]
                ax.add_patch(plt.Rectangle((col, y), 0.86, h, color=c["hex"], linewidth=0))
                y += h
    ax.set_xlim(-0.1, per_row)
    ax.set_ylim(len(names), -0.4)
    ax.set_yticks(np.arange(len(names)) + 0.42, names, fontfamily="monospace", fontsize=11)
    style(ax)
    ax.set_title("EVERY LOOK, IN ITS OWN COLORS", loc="left", fontsize=16, fontweight="bold", color=INK, pad=14)
    fig.text(0.125, 0.015, "Each bar is one look in running order; segment height = share of the garment.",
             fontsize=8, color=INK_MUTED)
    save(fig, "look_palettes.png")


def save(fig, name: str) -> None:
    out = ROOT / "figures" / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out.relative_to(ROOT)}")


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "missoni_dataset.csv")
    colors = look_colors(df)
    palette_strips(df, colors)
    look_palettes(df, colors)


if __name__ == "__main__":
    main()
