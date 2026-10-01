"""What makes a garment Missoni? An ablation test with the How-Missoni score.

Take Missoni runway looks, remove one ingredient at a time, and re-score each
against the archive with its own collection hidden:

- original      garment as photographed
- no colour     garment in greyscale
- light blur    Gaussian blur, radius 4 px: fine knit pattern softened
- heavy blur    radius 10 px: pattern gone, colour areas, drape and shading kept
- no pattern    garment flattened to its average colour

    python src/ablation.py          # re-use data/ablation_scores.csv if present
    python src/ablation.py --rerun  # recompute (5 looks per collection, ~2 min)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import how_missoni as hm
import segmentation
from chapter_charts import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE, save, titled
from clip_embeddings import NEUTRAL, embed_images
from visual_features import load_image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ablation_scores.csv"
CONDITIONS = {
    "original": "Original",
    "no_colour": "No colour (greyscale)",
    "blur_4": "Light blur (pattern softened)",
    "blur_10": "Heavy blur (pattern gone)",
    "no_pattern": "No pattern (one flat colour)",
}


def variants(full: np.ndarray) -> dict[str, np.ndarray]:
    gray = np.repeat(np.asarray(Image.fromarray(full).convert("L"))[..., None], 3, axis=2)
    return {
        "original": full,
        "no_colour": gray,
        "blur_4": np.asarray(Image.fromarray(full).filter(ImageFilter.GaussianBlur(4))),
        "blur_10": np.asarray(Image.fromarray(full).filter(ImageFilter.GaussianBlur(10))),
        "no_pattern": None,  # filled in once the garment mask is known
    }


def on_neutral(arr: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Same framing as clip_embeddings.garment_only, for a modified image."""
    ys, xs = np.nonzero(mask)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    canvas = np.where(mask[..., None], arr, np.array(NEUTRAL, dtype=np.uint8))[box].astype(np.uint8)
    crop = Image.fromarray(canvas)
    side = max(crop.size)
    square = Image.new("RGB", (side, side), NEUTRAL)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def run(per_collection: int = 5, seed: int = 0) -> pd.DataFrame:
    _, looks, _ = hm.reference()
    rng = np.random.default_rng(seed)
    sample = [i for _, g in looks.groupby("collection") for i in rng.choice(g.index, per_collection, replace=False)]
    rows = []
    for i in sample:
        look = looks.loc[i]
        full = load_image(str(ROOT / "data" / look["image_path"]))
        mask = segmentation.mask_of(segmentation.segment(full), segmentation.GARMENT)
        imgs = variants(full)
        imgs["no_pattern"] = np.broadcast_to(full[mask].mean(axis=0).astype(np.uint8), full.shape)
        embs = embed_images([on_neutral(imgs[k], mask) for k in CONDITIONS])
        row = {"look_id": look["look_id"]}
        for k, e in zip(CONDITIONS, embs):
            row[k] = round(hm.score_embedding(e, exclude=look["collection"])[0], 1)
        rows.append(row)
        print(f"  {look['look_id']}", end="\r")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    return df


def chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.27, right=0.95)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    keys = list(CONDITIONS)
    for y, k in enumerate(keys):
        ax.scatter(df[k], y + rng.uniform(-0.17, 0.17, len(df)), s=18, color=INK_MUTED, alpha=0.45, linewidths=0)
        med = df[k].median()
        ax.plot([med, med], [y - 0.3, y + 0.3], color="#2a78d6", linewidth=3, solid_capstyle="round")
        ax.text(med, y - 0.36, f"{med:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(range(len(keys)), [CONDITIONS[k] for k in keys])
    ax.set_ylim(len(keys) - 0.5, -0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("How-Missoni score (50 = a typical Missoni runway look)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_SECONDARY)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    titled(fig, "What makes it Missoni? The knit, not the colour",
           "Missoni looks re-scored with one ingredient removed. Blue bar = median; dots = individual looks.",
           f"{len(df)} looks, 5 per collection, each scored with its own collection hidden. "
           "Blurring keeps colour, shape and drape but removes the knit pattern.")
    save(fig, "ch06_ablation.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    df = run() if args.rerun or not OUT.exists() else pd.read_csv(OUT)
    print(df.drop(columns="look_id").median().round(1).to_dict())
    chart(df)


if __name__ == "__main__":
    main()
