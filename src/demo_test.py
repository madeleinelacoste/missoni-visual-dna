"""Does the How-Missoni score work on photos it has never seen?

Scores openly licensed museum/display photos from Wikimedia Commons: Missoni
pieces from outside the dataset vs other houses' prints and knits.
Download list and credits: data/test_images/sources.json (see README).

    python src/demo_test.py   # -> data/demo_test_scores.csv, figures/demo_test.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import how_missoni as hm
from chapter_charts import GRID, INK_MUTED, INK_SECONDARY, SURFACE, save, titled

ROOT = Path(__file__).resolve().parent.parent
LABELS = {
    "missoni_1973_barbados": "Missoni 1973 'Barbados' dress",
    "missoni_1970s_plaid_set": "Missoni 1970s plaid knit set",
    "missoni_2010_fishscale_dress": "Missoni 2010 fishscale dress*",
    "missoni_2010_coat": "Missoni 2010 coat",
    "missoni_2014_knit_dress": "Missoni 2014 knit dress",
    "pucci_1960s_silk_jersey": "Pucci 1960s silk print",
    "pucci_1970s_cocktail": "Pucci 1970s print dress",
    "pucci_1965_knit_fringe": "Pucci 1965 silk knit, fringe",
    "rykiel_1986_knit_set": "Sonia Rykiel 1986 striped knit",
    "french_1927_wool_knit": "French 1927 chevron wool knit",
    "aran_cardigan": "Aran cable cardigan",
    "win_patterned_sweater": "'WIN' patterned sweater",
}


def score_all() -> pd.DataFrame:
    meta = json.loads((ROOT / "data" / "test_images" / "sources.json").read_text())
    rows = []
    for m in meta:
        r = hm.analyse(ROOT / "data" / "test_images" / f"{m['file']}.jpg")
        rows.append({"file": m["file"], "group": m["group"], "score": round(r.score, 1),
                     "sim": round(r.similarity, 3), "closest": r.closest_collection, "garment": r.garment_found})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "demo_test_scores.csv", index=False)
    return df


def auc(df: pd.DataFrame) -> float:
    """Probability that a random Missoni piece outscores a random non-Missoni piece."""
    a = df.loc[df["group"] == "missoni", "score"].to_numpy()
    b = df.loc[df["group"] == "other", "score"].to_numpy()
    return float(((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])).mean())


def chart(df: pd.DataFrame) -> None:
    """Ranked dot plot: one row per photo, coloured by group."""
    colors = {"missoni": "#2a78d6", "other": "#eb6834"}
    names = {"missoni": "Missoni, not in the dataset", "other": "Other houses & knits"}
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    fig.subplots_adjust(top=0.78, bottom=0.11, left=0.3, right=0.95)
    ax.set_facecolor(SURFACE)
    for y, r in df.iterrows():
        ax.plot([0, r["score"]], [y, y], color=GRID, linewidth=1.5, zorder=1)
        ax.scatter(r["score"], y, s=70, color=colors[r["group"]], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.text(r["score"] + 1.5, y, f"{r['score']:.0f}", va="center", fontsize=8.5, color=INK_SECONDARY)
    for g, c in colors.items():
        ax.scatter([], [], s=70, color=c, label=f"{names[g]} (median {df[df.group == g].score.median():.0f})")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK_SECONDARY)
    ax.axvline(50, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(50.5, -0.9, "typical Missoni runway look", fontsize=8, color=INK_MUTED)
    ax.set_yticks(range(len(df)), [LABELS[f] for f in df["file"]])
    ax.set_ylim(len(df) - 0.5, -1.2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("How-Missoni score")
    ax.tick_params(length=0, colors=INK_SECONDARY)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    titled(fig, "The score spots Missoni-like knits, not Missoni",
           f"12 museum and display photos the model has never seen. A Missoni piece outscores\n"
           f"a non-Missoni one {auc(df):.0%} of the time, but two chevron/stripe knits by others score like Missoni.",
           "*Garment only partly segmented (close-up shop-window photo). Photos: Wikimedia Commons, credits in README.")
    save(fig, "demo_test.png")


def main() -> None:
    path = ROOT / "data" / "demo_test_scores.csv"
    df = pd.read_csv(path) if path.exists() else score_all()
    print(df.to_string(index=False), f"\nAUC {auc(df):.2f}")
    chart(df)


if __name__ == "__main__":
    main()
