"""Charts for the README chapters 01–03.

    python src/chapter_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"
# Categorical slots 1–2 of the dataviz reference palette (validated colour-blind safe as a pair).
SEASON_COLOR = {"SS": "#2a78d6", "FW": "#eb6834"}
SEASON_NAME = {"SS": "Spring/Summer", "FW": "Fall/Winter"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "axes.titlecolor": INK,
})


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "missoni_dataset.csv")
    df["collection"] = df["year"].astype(str) + " " + df["season"]
    return df


def frame(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylabel(ylabel)


def titled(fig, title: str, subtitle: str, note: str) -> None:
    fig.text(0.06, 0.965, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.905, subtitle, fontsize=10, color=INK_SECONDARY, va="top")
    fig.text(0.06, 0.015, note, fontsize=8, color=INK_MUTED)


def dot_timeline(ax, df: pd.DataFrame, metric: str, legend: bool = True) -> None:
    """One dot per look (jittered around its year), a bar at each collection's median."""
    rng = np.random.default_rng(0)
    for season in SEASON_COLOR:  # fixed order: Spring/Summer first
        g = df[df["season"] == season]
        color = SEASON_COLOR[season]
        x = g["year"] + rng.uniform(-0.28, 0.28, len(g))
        ax.scatter(x, g[metric], s=22, color=color, alpha=0.55, linewidths=0, label=SEASON_NAME[season])
        for year, gy in g.groupby("year"):
            ax.plot([year - 0.42, year + 0.42], [gy[metric].median()] * 2, color=color, linewidth=2.5,
                    solid_capstyle="round")
    years = sorted(df["year"].unique())
    ax.set_xticks(years, [str(y) for y in years])
    ax.set_xlim(min(years) - 1, max(years) + 1)
    if legend:
        ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_SECONDARY,
                  handletextpad=0.3, markerscale=1.4)


def save(fig, name: str) -> None:
    out = ROOT / "figures" / name
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out.relative_to(ROOT)}")


def ch01_colorfulness(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.08, right=0.97)
    frame(ax, "Colorfulness  (Hasler–Süsstrunk)")
    dot_timeline(ax, df, "colorfulness")
    ax.set_ylim(0, None)
    titled(fig, "01 — Missoni's color peaked in 2008, then cooled",
           "Colorfulness of each runway look's garment. Bars mark each collection's median.",
           "Dots = individual looks (15 per collection). ~15 slightly colorful · ~45 moderately · ~80 highly colorful.")
    save(fig, "ch01_colorfulness.png")


def ch02_pattern(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, facecolor=SURFACE)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.97, hspace=0.18)
    frame(axes[0], "Pattern complexity\n(share of edge pixels)")
    dot_timeline(axes[0], df, "pattern_complexity")
    frame(axes[1], "Edge regularity\n(higher = more striped)")
    dot_timeline(axes[1], df, "edge_regularity", legend=False)
    for ax in axes:
        ax.set_ylim(0, None)
    titled(fig, "02 — Pattern DNA: 2016 FW is the busiest, most striped show",
           "Top: how much of the garment is covered by pattern edges. Bottom: how aligned those edges are.",
           "Measured on segmented garment pixels only, with the garment outline excluded.")
    save(fig, "ch02_pattern.png")


def ch03_complexity(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.08, right=0.97)
    frame(ax, "Pattern complexity")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.scatter(df["hue_diversity"], df["pattern_complexity"], s=18, color=INK_MUTED, alpha=0.35, linewidths=0)
    med = df.groupby(["collection", "season"])[["hue_diversity", "pattern_complexity"]].median().reset_index()
    for _, r in med.iterrows():
        ax.scatter(r["hue_diversity"], r["pattern_complexity"], s=70, color=SEASON_COLOR[r["season"]],
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        # Hand-placed where labels would otherwise collide.
        offset = {"2001 FW": (7, -13), "2008 SS": (-58, 6)}.get(r["collection"], (7, 4))
        ax.annotate(r["collection"], (r["hue_diversity"], r["pattern_complexity"]), xytext=offset,
                    textcoords="offset points", fontsize=9, color=INK, fontfamily="monospace")
    for season, color in SEASON_COLOR.items():
        ax.scatter([], [], s=70, color=color, label=f"{SEASON_NAME[season]} median")
    ax.scatter([], [], s=18, color=INK_MUTED, alpha=0.5, label="Individual look")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    ax.set_xlabel("Hue diversity  (0 = one hue · 1 = full spectrum)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    titled(fig, "03 — 2016 FW and 2003 SS are the most complex; 2011 FW the plainest",
           "Each collection's median look. Up and right = more hues and more pattern.",
           "Hue diversity = entropy of the hue histogram over saturated garment pixels.")
    save(fig, "ch03_complexity.png")


def season_skin(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    fig.subplots_adjust(top=0.79, bottom=0.12, left=0.08, right=0.97)
    frame(ax, "Skin share of visible body")
    dot_timeline(ax, df, "skin_share")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    titled(fig, "Spring shows skin, except in 2019",
           "Share of the model's visible body that is skin rather than clothing, per look.",
           "From the segmentation model's face, arm and leg labels vs. garment labels. The face is included, so a fully covered look still scores ~5%.")
    save(fig, "skin_share.png")


def main() -> None:
    df = load()
    ch01_colorfulness(df)
    ch02_pattern(df)
    ch03_complexity(df)
    season_skin(df)


if __name__ == "__main__":
    main()
