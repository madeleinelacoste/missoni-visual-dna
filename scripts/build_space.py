"""Assemble a Hugging Face Space for the How-Missoni demo in build/space/.

The Space contains the app, the code it needs and the pre-computed CLIP
embeddings of the 135 Missoni looks. It contains no runway photos: matches
are listed with links to the archive lookbooks. Model weights download on
first start.

    python scripts/build_space.py
    # then upload build/space/ to a new Gradio Space (see build/space/README.md)
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "space"

FILES = [
    "app.py",
    "src/how_missoni.py", "src/clip_embeddings.py", "src/segmentation.py",
    "src/visual_features.py", "src/chapter_charts.py",
    "data/looks.csv", "data/embeddings_clip.csv",
]
REQUIREMENTS = ["gradio", "onnxruntime", "numpy", "pandas", "pillow", "scikit-learn", "matplotlib"]
SPACE_README = """---
title: How Missoni Is This?
emoji: 🧶
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: How close is an outfit to the Missoni runway archive?
---

# How Missoni Is This?

Upload an outfit photo. The garment is segmented (SegFormer-B2), embedded with CLIP and
compared with 135 Missoni runway looks from 2001–2023. The score is calibrated so that
**50 = a typical Missoni runway look**.

It mostly recognizes **Missoni-like knit pattern**, not the brand. For fun and learning; it cannot
authenticate anything. Full project, method and findings:
[github.com/madeleinelacoste/missoni-visual-dna](https://github.com/madeleinelacoste/missoni-visual-dna)

Models: SegFormer-B2 clothes (mattmdjaga, NVIDIA SegFormer licence, non-commercial) and
CLIP ViT-B/32 (OpenAI, MIT; Xenova ONNX export). No runway photos are hosted here; matches
link to the [Archivio Missoni](https://www.archiviomissoni.org) lookbooks.
"""


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for f in FILES:
        dest = OUT / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    (OUT / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n")
    (OUT / "README.md").write_text(SPACE_README)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{len(FILES) + 2} files, {size:.1f} MB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
