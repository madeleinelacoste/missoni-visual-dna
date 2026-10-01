"""Turn an Archivio Missoni lookbook PDF into look images + rows in looks.csv.

Step 1 — render every page and make a contact sheet to review:

    python src/import_lookbook.py pages data/lookbooks/2005_ESTATE-DONNA.pdf

Step 2 — import an evenly spaced sample of looks. Lookbooks mix full-length
runway shots with close-ups (bags, shoes, fabric), half-body portraits and
back views; the segmentation model keeps only full-length, front-facing looks.
`--first/--last` bound the pages to consider (e.g. skip the cover);
`--skip` drops any pages the filter gets wrong.

    python src/import_lookbook.py import data/lookbooks/2005_ESTATE-DONNA.pdf \\
        --year 2005 --season SS --era "Angela Missoni" --first 2 --last 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw

import segmentation

DATA = Path(__file__).resolve().parent.parent / "data"
SOURCE_BASE = "https://www.archiviomissoni.org/wp-content/uploads/"
TARGET_HEIGHT = 1200  # px; enough detail for knit patterns, small enough to be quick


def pages_dir(pdf: Path) -> Path:
    return pdf.with_suffix("")  # data/lookbooks/2005_ESTATE-DONNA/


def render_pages(pdf: Path) -> list[Path]:
    out = pages_dir(pdf)
    out.mkdir(exist_ok=True)
    paths = []
    for i, page in enumerate(pymupdf.open(pdf), start=1):
        path = out / f"p{i:03d}.jpg"
        if not path.exists():
            zoom = TARGET_HEIGHT / page.rect.height
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).pil_save(path, quality=92)
        paths.append(path)
    return paths


def contact_sheet(paths: list[Path], out: Path, cols: int = 10) -> None:
    tw, th = 140, 230
    sheet = Image.new("RGB", (tw * cols, (th + 16) * ((len(paths) + cols - 1) // cols)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = Image.open(p)
        im.thumbnail((tw, th))
        x, y = (i % cols) * tw, (i // cols) * (th + 16)
        sheet.paste(im, (x + (tw - im.width) // 2, y))
        draw.text((x + 4, y + th + 2), str(i + 1), fill="black")
    sheet.save(out, quality=85)


def is_full_look(labels: np.ndarray) -> bool:
    """Full-length, front-facing runway shot: face visible, and the body runs
    from the top third of the frame down to legs/shoes in the bottom quarter."""
    def rows(names):
        return np.nonzero(segmentation.mask_of(labels, names).any(axis=1))[0]

    face = rows(["face"])
    feet = rows(["left_leg", "right_leg", "left_shoe", "right_shoe"])
    garment = segmentation.mask_of(labels, segmentation.GARMENT)
    h = labels.shape[0]
    return (
        segmentation.mask_of(labels, ["face"]).mean() > 0.002
        and len(feet) > 0
        and face.min() < h / 3
        and feet.max() > 0.75 * h
        and garment.mean() > 0.03
    )


def full_look_pages(pdf: Path) -> list[int]:
    """Pages that are full-length looks, cached in pages/looks.json after the first run."""
    cache = pages_dir(pdf) / "looks.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pages = []
    for path in render_pages(pdf):
        img = np.asarray(Image.open(path).convert("RGB"))
        small = Image.fromarray(img)
        small.thumbnail((512, 512))
        if is_full_look(segmentation.segment(np.asarray(small))):
            pages.append(int(path.stem[1:]))
    cache.write_text(json.dumps(pages))
    return pages


def even_sample(pages: list[int], n: int) -> list[int]:
    idx = np.linspace(0, len(pages) - 1, min(n, len(pages))).round().astype(int)
    return [pages[i] for i in idx]


def import_looks(pdf: Path, year: int, season: str, era: str, first: int, last: int, skip: list[int], n: int) -> None:
    look_pages = [p for p in full_look_pages(pdf) if first <= p <= last and p not in skip]
    rel = pdf.relative_to(DATA / "lookbooks")
    # 2019+ lookbooks live in a dated upload folder on the archive site.
    url = SOURCE_BASE + ("2024/01/" if year >= 2019 else "") + str(rel)

    rows = []
    for page in even_sample(look_pages, n):
        # Position among the lookbook's full-length looks (≈ running order of the show).
        look = look_pages.index(page) + 1
        look_id = f"{year}_{season}_{look:03d}"
        Image.open(pages_dir(pdf) / f"p{page:03d}.jpg").save(DATA / "images" / f"{look_id}.jpg", quality=95)
        rows.append({
            "look_id": look_id, "year": year, "season": season, "look_number": look,
            "designer_era": era, "source": "Archivio Missoni", "source_url": f"{url}#page={page}",
            "image_path": f"images/{look_id}.jpg",
        })

    looks = pd.read_csv(DATA / "looks.csv")
    looks = looks[~looks["look_id"].isin([r["look_id"] for r in rows])]
    looks = pd.concat([looks, pd.DataFrame(rows)], ignore_index=True).sort_values(["year", "season", "look_number"])
    looks.to_csv(DATA / "looks.csv", index=False)
    print(f"{len(rows)} looks from {len(look_pages)} -> data/images/, looks.csv now has {len(looks)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages")
    p.add_argument("pdf", type=Path)
    i = sub.add_parser("import")
    i.add_argument("pdf", type=Path)
    i.add_argument("--year", type=int, required=True)
    i.add_argument("--season", required=True, choices=["SS", "FW"])
    i.add_argument("--era", required=True)
    i.add_argument("--first", type=int, default=1)
    i.add_argument("--last", type=int, default=10_000)
    i.add_argument("--skip", type=int, nargs="*", default=[])
    i.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    pdf = args.pdf.resolve()
    if args.cmd == "pages":
        paths = render_pages(pdf)
        sheet = pages_dir(pdf) / "contact_sheet.jpg"
        contact_sheet(paths, sheet)
        print(f"{len(paths)} pages -> {sheet.relative_to(DATA.parent)}")
    else:
        import_looks(pdf, args.year, args.season, args.era, args.first, args.last, args.skip, args.n)


if __name__ == "__main__":
    main()
