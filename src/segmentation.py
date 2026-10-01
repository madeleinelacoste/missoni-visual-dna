"""Clothing segmentation: label every pixel of a runway photo as garment, skin, hair or background.

Model: SegFormer-B2 fine-tuned on the ATR human-parsing dataset
(huggingface.co/mattmdjaga/segformer_b2_clothes), run through ONNX Runtime so
no PyTorch install is needed. Licence: NVIDIA SegFormer licence
(non-commercial / research use).
"""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_URL = "https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "segformer_b2_clothes.onnx"
INPUT_SIZE = 512

LABELS = [
    "background", "hat", "hair", "sunglasses", "upper_clothes", "skirt", "pants", "dress",
    "belt", "left_shoe", "right_shoe", "face", "left_leg", "right_leg", "left_arm",
    "right_arm", "bag", "scarf",
]
# What counts as "the garment" for colour analysis. Shoes, bags and hats are
# styling, not the collection's knitwear, so they're left out.
GARMENT = ["upper_clothes", "skirt", "pants", "dress", "belt", "scarf"]
SKIN = ["face", "left_leg", "right_leg", "left_arm", "right_arm"]


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        print(f"Downloading segmentation model (110 MB) -> {MODEL_PATH.name}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def segment(img: np.ndarray) -> np.ndarray:
    """RGB image (h, w, 3) -> label map (h, w) of indices into LABELS."""
    h, w = img.shape[:2]
    x = np.asarray(Image.fromarray(img).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR), dtype=np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    logits = _session().run(None, {"pixel_values": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    # The model predicts at 1/4 resolution: upsample each class's scores, then pick the best.
    up = np.stack([np.asarray(Image.fromarray(c).resize((w, h), Image.BILINEAR)) for c in logits])
    return up.argmax(axis=0)


def mask_of(labels: np.ndarray, names: list[str]) -> np.ndarray:
    return np.isin(labels, [LABELS.index(n) for n in names])


def silhouette(labels: np.ndarray, min_share: float = 0.15) -> str:
    """Rough silhouette from which garment classes cover the outfit."""
    garment_px = mask_of(labels, GARMENT).sum()
    if garment_px == 0:
        return "unknown"
    share = {n: (labels == LABELS.index(n)).sum() / garment_px for n in ["upper_clothes", "skirt", "pants", "dress"]}
    present = {n for n, s in share.items() if s >= min_share}
    if "dress" in present and not present & {"skirt", "pants"}:
        return "dress"
    if "pants" in present:
        return "top+trousers" if "upper_clothes" in present else "trousers"
    if "skirt" in present:
        return "top+skirt" if "upper_clothes" in present else "skirt"
    return max(share, key=share.get).replace("upper_clothes", "top")
