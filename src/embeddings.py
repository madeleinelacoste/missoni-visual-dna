"""Visual fingerprints ("embeddings") for every look.

A hand-built embedding from the segmented garment pixels:

- colour: a CIELAB histogram (8x8 bins over the a*/b* colour plane + 5
  lightness bins), square-rooted so distances behave like the Hellinger
  distance between colour distributions;
- texture: pattern complexity, edge regularity and an 8-bin histogram of edge
  directions (stripes, zigzags and florals have different direction profiles).

    python src/embeddings.py   # -> data/embeddings_handcrafted.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from visual_features import edge_regularity, isolate_garment, load_image, pattern_complexity, to_lab

DATA = Path(__file__).resolve().parent.parent / "data"
AB_BINS, AB_RANGE, L_BINS = 8, 60, 5


def color_histogram(fabric_rgb: np.ndarray) -> np.ndarray:
    lab = to_lab(fabric_rgb)
    ab, _, _ = np.histogram2d(lab[:, 1], lab[:, 2], bins=AB_BINS, range=[[-AB_RANGE, AB_RANGE]] * 2)
    light, _ = np.histogram(lab[:, 0], bins=L_BINS, range=(0, 100))
    return np.sqrt(np.concatenate([ab.ravel() / ab.sum(), light / light.sum()]))


def edge_directions(img: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray:
    gray = np.asarray(Image.fromarray(img).convert("L").filter(ImageFilter.GaussianBlur(1))).astype(float)
    gx, gy = np.diff(gray, axis=1)[:-1, :], np.diff(gray, axis=0)[:, :-1]
    mag = np.hypot(gx, gy)
    inside = mask[:-1, :-1] & (mag > 10)
    if inside.sum() < 50:
        return np.zeros(bins)
    angle = np.mod(np.arctan2(gy[inside], gx[inside]), np.pi)
    hist, _ = np.histogram(angle, bins=bins, range=(0, np.pi), weights=mag[inside])
    return np.sqrt(hist / hist.sum())


def embed(path: str) -> np.ndarray:
    img, mask, _ = isolate_garment(load_image(path))
    fabric = img[mask] if mask.sum() > 100 else img.reshape(-1, 3)
    texture = [pattern_complexity(img, mask), edge_regularity(img, mask)]
    return np.concatenate([color_histogram(fabric), texture, edge_directions(img, mask)])


def main() -> None:
    looks = pd.read_csv(DATA / "looks.csv")
    rows = []
    for _, look in looks.iterrows():
        rows.append(embed(str(DATA / look["image_path"])))
        print(f"  {look['look_id']}", end="\r")
    n_color = AB_BINS * AB_BINS + L_BINS
    cols = [f"color_{i}" for i in range(n_color)] + ["pattern_complexity", "edge_regularity"] + [
        f"edge_dir_{i}" for i in range(8)
    ]
    out = pd.DataFrame(rows, columns=cols).round(5)
    out.insert(0, "look_id", looks["look_id"])
    out.to_csv(DATA / "embeddings_handcrafted.csv", index=False)
    print(f"\n{len(out)} looks x {len(cols)} dims -> data/embeddings_handcrafted.csv")


if __name__ == "__main__":
    main()
