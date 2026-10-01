"""How Missoni Is This? Score any outfit photo against the Missoni runway dataset.

Not an authentication tool, just a demonstration of image embeddings and
similarity. The photo's garment is segmented, embedded with CLIP and compared
with the 135 garment-only Missoni embeddings.

The score is calibrated against Missoni itself: each archive look is scored
against the archive *with its own collection hidden* (as an unseen image would
be), and a new image's score is the share of those looks it matches or beats.
So 50 means "as Missoni as a typical Missoni runway look".

    python src/how_missoni.py path/to/photo.jpg   # -> results/<name>_card.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import segmentation
from clip_embeddings import embed_images, garment_only
from visual_features import dominant_colors, isolate_garment

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5
DUPLICATE_SIMILARITY = 0.98  # an archive image matching itself; ignored so it's scored like an unseen one


@dataclass
class Result:
    score: float  # 0–100 percentile against Missoni's own looks
    similarity: float  # mean cosine similarity to the TOP_K closest Missoni looks
    closest_collection: str
    collection_votes: dict[str, float]
    neighbours: pd.DataFrame  # closest Missoni looks
    palette: list[tuple[str, float]]  # the photo's garment colours
    garment: Image.Image  # what the model actually looked at
    garment_found: bool


@lru_cache(maxsize=1)
def reference() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    emb = pd.read_csv(ROOT / "data" / "embeddings_clip.csv")
    looks = pd.read_csv(ROOT / "data" / "looks.csv").set_index("look_id").loc[emb["look_id"]].reset_index()
    looks["collection"] = looks["year"].astype(str) + " " + looks["season"]
    E = emb.drop(columns="look_id").to_numpy()
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    # Calibration: each look's top-k similarity with its own collection hidden.
    sims = E @ E.T
    same = looks["collection"].to_numpy()[:, None] == looks["collection"].to_numpy()[None, :]
    sims[same] = -np.inf
    calibration = np.sort(sims, axis=1)[:, -TOP_K:].mean(axis=1)
    return E, looks, calibration


def score_embedding(e: np.ndarray, exclude: str | None = None) -> tuple[float, float, pd.DataFrame, dict]:
    E, looks, calibration = reference()
    sims = E @ e
    keep = np.ones(len(looks), dtype=bool) if exclude is None else (looks["collection"] != exclude).to_numpy()
    sims = np.where(keep & (sims < DUPLICATE_SIMILARITY), sims, -np.inf)
    top = np.argsort(sims)[::-1][:TOP_K]
    similarity = float(sims[top].mean())
    cal = calibration if exclude is None else calibration[(looks["collection"] != exclude).to_numpy()]
    score = 100 * float((cal <= similarity).mean())
    neighbours = looks.iloc[top][["look_id", "collection", "image_path", "source_url"]].assign(similarity=sims[top])
    votes = neighbours.groupby("collection")["similarity"].sum()
    return score, similarity, neighbours.reset_index(drop=True), (votes / votes.sum()).sort_values(ascending=False).to_dict()


def analyse(path: str | Path) -> Result:
    full = np.asarray(Image.open(path).convert("RGB"))
    labels = segmentation.segment(full)
    found = segmentation.mask_of(labels, segmentation.GARMENT).mean() >= 0.01
    garment = garment_only(full)
    e = embed_images([garment])[0]
    score, similarity, neighbours, votes = score_embedding(e)
    img, mask, _ = isolate_garment(full)
    fabric = img[mask] if mask.sum() > 100 else img.reshape(-1, 3)
    return Result(score, similarity, next(iter(votes)), votes, neighbours, dominant_colors(fabric), garment, found)


def verdict(score: float, garment_found: bool = True) -> str:
    if not garment_found:
        return "No garment detected"
    if score >= 60:
        return "Very Missoni"
    if score >= 30:
        return "Missoni-adjacent"
    if score >= 10:
        return "A hint of Missoni"
    return "Not very Missoni"


def card(result: Result, out: Path, show_neighbours: bool = True) -> None:
    """A one-image summary. Neighbour photos are archive images, so cards that
    include them are for local use only (results/ is git-ignored)."""
    import matplotlib.pyplot as plt

    from chapter_charts import INK, INK_MUTED, INK_SECONDARY, SURFACE

    n = len(result.neighbours) if show_neighbours else 0
    fig = plt.figure(figsize=(11, 5.6), facecolor=SURFACE)
    ax_img = fig.add_axes([0.03, 0.08, 0.28, 0.84])
    ax_img.imshow(result.garment)
    ax_img.axis("off")
    fig.text(0.35, 0.86, "HOW MISSONI IS THIS?", fontsize=11, color=INK_MUTED, fontweight="bold")
    fig.text(0.35, 0.68, f"{result.score:.0f}", fontsize=58, color=INK, fontweight="bold")
    fig.text(0.47, 0.71, f"/ 100   {verdict(result.score, result.garment_found)}", fontsize=15, color=INK_SECONDARY)
    fig.text(0.35, 0.62, f"As Missoni as {result.score:.0f}% of Missoni's own runway looks.", fontsize=10,
             color=INK_SECONDARY)
    fig.text(0.35, 0.565, f"Closest collection: {result.closest_collection}", fontsize=12, color=INK,
             fontfamily="monospace")
    ax_pal = fig.add_axes([0.35, 0.44, 0.6, 0.06])
    x = 0.0
    for hex_, share in result.palette:
        ax_pal.add_patch(plt.Rectangle((x, 0), share - 0.004, 1, color=hex_, linewidth=0))
        x += share
    ax_pal.set_xlim(0, 1)
    ax_pal.axis("off")
    fig.text(0.35, 0.51, "garment palette", fontsize=8, color=INK_MUTED)
    for i, row in enumerate(result.neighbours.head(n).itertuples()):
        ax = fig.add_axes([0.35 + i * 0.122, 0.04, 0.11, 0.3])
        nb = Image.open(ROOT / "data" / row.image_path)
        nb.thumbnail((300, 450))
        ax.imshow(nb)
        ax.axis("off")
        ax.set_title(f"{row.collection}\n{row.similarity:.2f}", fontsize=8, color=INK_SECONDARY, fontfamily="monospace")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.images:
        r = analyse(path)
        out = ROOT / "results" / f"{path.stem}_card.png"
        card(r, out)
        print(f"{path.name}: {r.score:.0f}/100 ({verdict(r.score, r.garment_found)}) · closest {r.closest_collection} · "
              f"similarity {r.similarity:.3f} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
