"""Visual features for a single runway image.

Everything here works on plain NumPy arrays so each step is easy to inspect
in a notebook: load -> crop to the garment -> measure colour and pattern.
"""

from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans

# Runway photos are tall, with the model roughly centred. This box (as
# fractions of width/height) keeps the torso-to-knee area and drops most of
# the background, floor and head. Tune it once you've looked at your images.
GARMENT_BOX = (0.30, 0.18, 0.70, 0.75)  # left, top, right, bottom

# Downscale before analysis: colour statistics barely change and it's ~50x faster.
ANALYSIS_SIZE = 256


def load_image(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 array of shape (h, w, 3)."""
    return np.asarray(Image.open(path).convert("RGB"))


def crop_garment(img: np.ndarray, box: tuple[float, float, float, float] = GARMENT_BOX) -> np.ndarray:
    h, w = img.shape[:2]
    left, top, right, bottom = box
    return img[int(top * h):int(bottom * h), int(left * w):int(right * w)]


def downscale(img: np.ndarray, size: int = ANALYSIS_SIZE) -> np.ndarray:
    pil = Image.fromarray(img)
    pil.thumbnail((size, size))
    return np.asarray(pil)


def dominant_colors(img: np.ndarray, k: int = 5, seed: int = 0) -> list[tuple[str, float]]:
    """K-means on pixels -> [(hex, share), ...] sorted by share, largest first."""
    pixels = img.reshape(-1, 3).astype(float)
    k = min(k, len(np.unique(pixels, axis=0)))  # a flat solid has fewer than k colours
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(pixels)
    shares = np.bincount(km.labels_, minlength=k) / len(pixels)
    order = np.argsort(-shares)
    return [(rgb_to_hex(km.cluster_centers_[i]), float(shares[i])) for i in order]


def colorfulness(img: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric.

    Roughly: 0 = greyscale, ~15 slightly colourful, ~45 moderately,
    ~80 highly, 100+ extremely colourful.
    """
    r, g, b = (img[..., i].astype(float) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.hypot(rg.std(), yb.std())
    mean = np.hypot(rg.mean(), yb.mean())
    return float(std + 0.3 * mean)


def hsv_stats(img: np.ndarray) -> dict[str, float]:
    """Mean brightness and saturation (0–1) from HSV."""
    rgb = img.reshape(-1, 3).astype(float) / 255
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    saturation = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0)
    return {"brightness": float(mx.mean()), "saturation": float(saturation.mean())}


def hue_diversity(img: np.ndarray, bins: int = 12, min_saturation: float = 0.2) -> float:
    """Shannon entropy of the hue histogram, normalised to 0–1.

    Only reasonably saturated pixels count, so a black-and-white look scores
    ~0 and a full rainbow knit scores close to 1.
    """
    rgb = img.reshape(-1, 3).astype(float) / 255
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in rgb[:: max(1, len(rgb) // 4000)]])
    hues = hsv[hsv[:, 1] >= min_saturation, 0]
    if len(hues) == 0:
        return 0.0
    counts, _ = np.histogram(hues, bins=bins, range=(0, 1))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(bins))


def pattern_complexity(img: np.ndarray) -> float:
    """Edge density: share of pixels with a strong luminance gradient (0–1).

    Plain fabric -> low; tight zigzag / multi-stripe knit -> high.
    """
    gray = img.astype(float) @ np.array([0.299, 0.587, 0.114])
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    return float((np.hypot(gx, gy) > 25).mean())


def edge_regularity(img: np.ndarray, bins: int = 8) -> float:
    """How few directions the pattern's edges run in (0 = every direction, 1 = one direction).

    Stripes score ~1, zigzags ~0.3 (two directions), florals, abstract prints
    and noise-like textures ~0.
    """
    # A light blur stops JPEG artefacts and pixel staircases posing as edges.
    gray = np.asarray(Image.fromarray(img).convert("L").filter(ImageFilter.GaussianBlur(1))).astype(float)
    gx = np.diff(gray, axis=1)[:-1, :]
    gy = np.diff(gray, axis=0)[:, :-1]
    mag = np.hypot(gx, gy)
    strong = mag > max(np.percentile(mag, 80), 5)
    if strong.sum() < 50:
        return 0.0
    # Edge orientation in [0, pi): an edge and its reverse count as the same.
    angle = np.mod(np.arctan2(gy[strong], gx[strong]), np.pi)
    counts, _ = np.histogram(angle, bins=bins, range=(0, np.pi), weights=mag[strong])
    p = counts[counts > 0] / counts.sum()
    return float(1 + (p * np.log(p)).sum() / np.log(bins))


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(c)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def extract(path: str, k: int = 5) -> dict:
    """All features for one image, flattened into a single dict (one CSV row)."""
    img = downscale(crop_garment(load_image(path)))
    row: dict = {}
    for i, (hex_, share) in enumerate(dominant_colors(img, k=k), start=1):
        row[f"color_{i}"] = hex_
        row[f"color_{i}_share"] = round(share, 4)
    row["colorfulness"] = round(colorfulness(img), 2)
    row.update({key: round(v, 4) for key, v in hsv_stats(img).items()})
    row["hue_diversity"] = round(hue_diversity(img), 4)
    row["pattern_complexity"] = round(pattern_complexity(img), 4)
    row["edge_regularity"] = round(edge_regularity(img), 4)
    return row
