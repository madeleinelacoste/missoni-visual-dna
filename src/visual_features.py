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
GARMENT_BOX = (0.37, 0.20, 0.63, 0.66)  # left, top, right, bottom

# Where the model's face sits in a runway photo; used to sample her skin tone.
FACE_BOX = (0.44, 0.10, 0.56, 0.18)

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


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (uint8, any shape ending in 3) -> CIELAB, where distances match perceived colour difference."""
    c = rgb.astype(float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def skin_like(img: np.ndarray) -> np.ndarray:
    """Broad YCbCr skin rule (Chai & Ngan). Catches all skin but also many beiges and yellows."""
    ycc = np.asarray(Image.fromarray(img).convert("YCbCr")).astype(int)
    cb, cr = ycc[..., 1], ycc[..., 2]
    return (cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 128)


def skin_mask(full: np.ndarray, crop: np.ndarray, max_distance: float = 8.0) -> np.ndarray:
    """Pixels of `crop` that are this model's skin rather than fabric.

    The broad rule alone eats sand, peach and yellow fabrics, so instead we
    sample the model's own skin tone from her face and keep only pixels close
    to it in CIELAB. Lightness is down-weighted so the same skin in shadow or
    highlight still matches, while hue and chroma must be close.
    """
    face = crop_garment(full, FACE_BOX)
    face_skin = face[skin_like(face)]
    if len(face_skin) < 30:  # face hidden or out of frame: don't guess
        return np.zeros(crop.shape[:2], dtype=bool)
    ref = to_lab(face_skin).mean(axis=0)
    d = (to_lab(crop) - ref) / np.array([3.0, 1.0, 1.0])
    return (np.linalg.norm(d, axis=-1) < max_distance) & skin_like(crop)


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
    """All features for one image, flattened into a single dict (one CSV row).

    Colour features use fabric pixels only (skin removed); pattern features
    use the whole crop, since they need the 2-D layout of the image.
    """
    full = load_image(path)
    img = downscale(crop_garment(full))
    skin = skin_mask(full, img)
    # A look that is almost all skin (e.g. swimwear) still needs some pixels to measure.
    fabric = img[~skin] if (~skin).mean() > 0.05 else img.reshape(-1, 3)

    row: dict = {"skin_share": round(float(skin.mean()), 4)}
    for i, (hex_, share) in enumerate(dominant_colors(fabric, k=k), start=1):
        row[f"color_{i}"] = hex_
        row[f"color_{i}_share"] = round(share, 4)
    row["colorfulness"] = round(colorfulness(fabric), 2)
    row.update({key: round(v, 4) for key, v in hsv_stats(fabric).items()})
    row["hue_diversity"] = round(hue_diversity(fabric), 4)
    row["pattern_complexity"] = round(pattern_complexity(img), 4)
    row["edge_regularity"] = round(edge_regularity(img), 4)
    return row
