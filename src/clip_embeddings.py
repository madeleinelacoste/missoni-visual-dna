"""CLIP image embeddings for every look.

CLIP (ViT-B/32, OpenAI, MIT licence; ONNX export from Xenova/clip-vit-base-patch32)
maps an image to a 512-number vector where visually and semantically similar
images sit close together.

Each runway show has its own venue, lighting and backdrop, so a model shown
the whole photo can "recognise" a collection from the room rather than the
clothes. The main embedding therefore uses the garment only: segmented, on a
neutral grey background. A full-frame version is computed too, purely to
measure how much the venue would have inflated the results.

    python src/clip_embeddings.py
    # -> data/embeddings_clip.csv, data/embeddings_clip_fullframe.csv
"""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import segmentation
from visual_features import load_image

ROOT = Path(__file__).resolve().parent.parent
MODEL_URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx"
MODEL_PATH = ROOT / "models" / "clip_vit_b32_vision.onnx"
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])
# Background for garment-only images: CLIP's mean colour, so it reads as "nothing".
NEUTRAL = tuple(int(round(v * 255)) for v in MEAN)


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        print(f"Downloading CLIP vision model (352 MB) -> {MODEL_PATH.name}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def _preprocess(img: Image.Image) -> np.ndarray:
    """CLIP preprocessing: shortest edge to 224 (bicubic), centre crop, normalise."""
    w, h = img.size
    scale = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.BICUBIC)
    left, top = (img.width - SIZE) // 2, (img.height - SIZE) // 2
    x = np.asarray(img.crop((left, top, left + SIZE, top + SIZE)), dtype=np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def embed_images(images: list[Image.Image]) -> np.ndarray:
    """-> (n, 512) unit-length embeddings."""
    batch = np.stack([_preprocess(im.convert("RGB")) for im in images])
    emb = _session().run(None, {"pixel_values": batch})[0]
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def garment_only(full: np.ndarray) -> Image.Image:
    """The segmented garment on a neutral background, padded to a square around it."""
    garment = segmentation.mask_of(segmentation.segment(full), segmentation.GARMENT)
    if garment.mean() < 0.01:  # nothing found: fall back to the whole photo
        return Image.fromarray(full)
    canvas = np.where(garment[..., None], full, np.array(NEUTRAL, dtype=np.uint8))
    ys, xs = np.nonzero(garment)
    crop = Image.fromarray(canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8))
    side = max(crop.size)
    square = Image.new("RGB", (side, side), NEUTRAL)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def main() -> None:
    looks = pd.read_csv(ROOT / "data" / "looks.csv")
    garment_rows, full_rows = [], []
    for _, look in looks.iterrows():
        full = load_image(str(ROOT / "data" / look["image_path"]))
        garment_rows.append(embed_images([garment_only(full)])[0])
        full_rows.append(embed_images([Image.fromarray(full)])[0])
        print(f"  {look['look_id']}", end="\r")
    for rows, name in [(garment_rows, "clip"), (full_rows, "clip_fullframe")]:
        out = pd.DataFrame(np.round(rows, 5), columns=[f"d{i}" for i in range(512)])
        out.insert(0, "look_id", looks["look_id"])
        out.to_csv(ROOT / "data" / f"embeddings_{name}.csv", index=False)
    print(f"\n{len(looks)} looks -> data/embeddings_clip.csv, data/embeddings_clip_fullframe.csv")


if __name__ == "__main__":
    main()
