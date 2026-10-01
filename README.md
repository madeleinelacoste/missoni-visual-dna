# MISSONI — Visual DNA

### Quantifying color, pattern & design across decades of Italian fashion

*A computer-vision study of how Missoni's visual identity has evolved across womenswear collections, using color extraction, image embeddings, clustering and visual similarity.*

![Missoni — years in color](figures/palette_strips.png)

---

## The question

**Can data and computer vision measure what makes a garment recognizably Missoni?**

The house is known for its zigzags, space-dyed stripes and dense color compositions. This project asks how much of that identity can be measured from runway images alone, and how it has changed from Ottavio and Rosita Missoni to the present day.

## Five investigations

| | chapter | question |
|---|---|---|
| 01 | **The Colors of Missoni** | How has the house palette changed by decade? |
| 02 | **Pattern DNA** | How often do zigzags, stripes and geometric patterns appear? |
| 03 | **Visual Complexity** | Are some eras more colorful or visually complex than others? |
| 04 | **Evolution vs. Consistency** | Which collections look most alike? |
| 05 | **Missoni's Eras** | Can unsupervised clustering find aesthetic periods without being told the year? |

## Method

```
runway image ──► segment garment ──► measure ──────────────────────► missoni_dataset.csv
                 (SegFormer-B2)     │
                                    ├─ silhouette (dress / top+skirt / top+trousers …)
                                    ├─ skin share (body shown vs. covered)
                                    ├─ dominant colors (k-means)
                                    ├─ colorfulness (Hasler–Süsstrunk)
                                    ├─ brightness · saturation
                                    ├─ hue diversity (hue entropy)
                                    ├─ pattern complexity (edge density)
                                    └─ edge regularity (stripe ≈ 1, zigzag ≈ 0.3, print ≈ 0)
```

| feature | what it captures | range |
|---|---|---|
| `silhouette_auto` | outfit shape, read from which garment classes the model finds | `dress`, `top+skirt`, `top+trousers`, … |
| `skin_share` | share of the visible body that is skin rather than clothing | 0–1 |
| `color_1…5` + `_share` | the five dominant colors and how much of the garment each covers | hex, 0–1 |
| `colorfulness` | overall chromatic intensity | 0 (grey) → 100+ (extremely colorful) |
| `brightness`, `saturation` | mean HSV value and saturation | 0–1 |
| `hue_diversity` | how many different hues appear | 0 (monochrome) → 1 (full spectrum) |
| `pattern_complexity` | share of pixels on a strong edge | 0 (plain) → high (dense knit pattern) |
| `edge_regularity` | how few directions the edges run in | 1 (stripes) · ~0.3 (zigzag) · ~0 (floral/abstract) |

### Isolating the garment

Every pixel is labelled as garment, skin, hair or background by **SegFormer-B2**, a transformer segmentation model fine-tuned for clothing ([mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes); NVIDIA SegFormer licence, non-commercial use). It runs locally with ONNX Runtime at about 0.7 s per look. Only garment pixels feed the color and pattern features, so the runway floor, background models and skin no longer distort the palette.

This replaced an earlier pipeline (fixed crop box plus a skin mask sampled from the model's face), which is kept as a fallback:

![Before/after: SS 2005 palette](figures/method_comparison_2005ss.png)

Pattern type is **hand-labelled on a subset** and used later to train and test a classifier.

## Data

One row per runway look. Sources: [Archivio Missoni](https://www.archiviomissoni.org) for the house's historical collections and Vogue Runway for recent seasons. Images stay local; the repo publishes only source links and the numbers derived from them. See [`data/README.md`](data/README.md).

**Creative directors** *(verify dates before publishing)*

| era | creative direction |
|---|---|
| 1953–1997 | Ottavio & Rosita Missoni |
| 1997–2021 | Angela Missoni |
| 2021–2023 | Alberto Caliri |
| 2023– | Filippo Grazioli, then successors |

## Reproduce

```bash
pip install -r requirements.txt
# 1. add images to data/images/ and one row per look to data/looks.csv
python src/extract_features.py    # -> data/missoni_dataset.csv (first run downloads the 110 MB model)
python src/palette_strips.py      # -> figures/palette_strips.png
```

## Roadmap

- [x] Feature extraction pipeline
- [x] Collection palette strips
- [x] Adaptive skin masking
- [x] Garment segmentation (SegFormer-B2)
- [x] First collection: SS 2005, 15 looks
- [x] Nine collections spanning 2001–2023 (135 looks), imported automatically from the archive lookbooks
- [ ] Pre-2001 collections (Ottavio & Rosita era) from other sources
- [ ] Chapters 01–03: palette, pattern and complexity by decade
- [ ] CLIP image embeddings, then PCA/UMAP map of every look (chapters 04–05)
- [ ] Pattern classifier trained on the hand-labelled subset
- [ ] *How Missoni Is This?* Upload an image and find the closest Missoni era (a demo of embeddings, not an authentication tool)

## Limitations

- The segmentation model was trained on everyday clothing, not runway knitwear. It treats swimwear as `top`, a jacket over a dress mostly as `dress`, and sheer fabric over skin as garment.
- Lookbook photography changes across the archive: scanned film in 2001–2003, then digital, with a different venue and lighting each season. Some cross-season color differences come from the camera rather than the clothes.
- The online archive starts at 2001, so the Ottavio & Rosita era is not yet covered. The SS 2019 show was co-ed and its lookbook includes menswear looks.
- A generic skin-color rule was tried for the fallback and rejected: it erased the beige, peach and yellow fabrics common in Missoni collections.
- Photography changes over 70 years (film stock, lighting, studio and runway shots), and this affects measured color. Treat cross-decade comparisons with care.
- Archive coverage is uneven, and early decades will have fewer looks.

---

*Part of a portfolio on data × culture × creativity — [madeleinelacoste](https://github.com/madeleinelacoste)*
