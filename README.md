# MISSONI — Visual DNA

### Quantifying color, pattern & design across decades of Italian fashion

*A computer-vision study of how Missoni's visual identity has evolved across womenswear collections, using color extraction, image embeddings, clustering and visual similarity.*

![Missoni — years in color](figures/palette_strips.png)

---

## The question

**Can data and computer vision measure what makes a garment recognizably Missoni?**

The house is known for its zigzags, space-dyed stripes and dense color compositions. This project asks how much of that identity can be measured from runway images alone, and how it has changed from Ottavio and Rosita Missoni to the present day.

## Findings so far

*9 collections, 2001–2023, 135 runway looks from the Archivio Missoni lookbooks.*

### 01 — The Colors of Missoni

*(Palette chart at the top of this page.)* Warm rust, camel and brown dominate the early 2000s. Teal and coral accents arrive in 2005–2008, followed by a shift to dusty mauve, grey and near-neutrals from 2011. **SS 2019 is the only collection with no vivid accent at all.**

![Colorfulness over time](figures/ch01_colorfulness.png)

Median colorfulness climbs from 42 (FW 2001) to 52 (SS 2008), then drops to 19–31 from 2011 on.

<details><summary>Every look in its own colors</summary>

![Every look's palette](figures/look_palettes.png)

</details>

### 02 — Pattern DNA

![Pattern complexity and regularity](figures/ch02_pattern.png)

FW 2016 is the busiest and most striped collection in the sample (median pattern complexity 0.49), which matches its heavy stripe and zigzag knits. FW 2011 and SS 2019 are the plainest.

![Pattern mix](figures/ch02b_pattern_mix.png)

Labelling each look's most prominent pattern ([definitions](data/README.md#pattern-labels); AI-assisted labels, reviewed by the author) shows the house cycling through patterns: prints dominate SS 2008 (10 of 15) and SS 2014 (9), stripes dominate FW 2016 (7), and **the zigzag returns in FW 2023 (7 of 15)**, its strongest showing in the sample.

![Pattern classifier](figures/ch02c_pattern_classifier.png)

Can a model learn those labels? A logistic regression on garment-only CLIP embeddings, tested on collections it never saw ([`src/pattern_analysis.py`](src/pattern_analysis.py)), gets **44% right vs 30%** for always guessing "print" (balanced accuracy 39% vs 16% chance, p < 0.01). Prints are easy (68%). The main failure is **zigzag vs stripe**: at CLIP's 224-pixel input a fine zigzag reads as a stripe, and many looks mix the two. Merging zigzag, stripe and space-dye into one "knit pattern" class raises accuracy to 64% (baseline 49%).

### 03 — Visual Complexity

![Hue diversity vs pattern complexity](figures/ch03_complexity.png)

Color variety and pattern density rise together: collections with more hues also tend to have more pattern edges. SS 2003 and FW 2016 sit at the complex end and FW 2011 at the minimal end.

### 04 — Evolution vs. Consistency

![Collection similarity](figures/ch04_similarity.png)

Each look gets a 77-number visual fingerprint: a CIELAB color histogram plus a histogram of edge directions, both computed on garment pixels only ([`src/embeddings.py`](src/embeddings.py)). The closest pair is **FW 2001 & SS 2003**, and the next closest is **FW 2023 & FW 2001**, which suggests the 2023 collection looked back to the house's early-2000s palette. SS 2005 is the outlier: its pale gold and turquoise chiffons are far from everything after it.

### 05 — Missoni's Eras

![Era map](figures/ch05_era_map.png)

Can the looks reveal their own periods? Partly:

- **Each show has a recognizable signature.** Given a look it hasn't seen, a 5-nearest-neighbour model names its collection **42% of the time vs 10% by chance** (permutation test, p = 0.005). Color carries this; texture alone reaches only 19%.
- **There is no single timeline.** Collection centres zigzag across the map instead of drifting in one direction, and neither main axis correlates with year (|r| < 0.2). In this data, Missoni moves between a few recurring looks rather than evolving steadily.

![Distinctiveness](figures/ch05_distinctiveness.png)

SS 2003 is the most recognizable collection (73%), followed by FW 2011 and SS 2014 (67%). FW 2001 and SS 2008 are the least distinct (13%), sharing their vocabulary with neighbouring seasons.

### 05b — What does a neural network see?

The same tests were repeated with **CLIP** (ViT-B/32), a general-purpose vision model that turns an image into a 512-number embedding ([`src/clip_embeddings.py`](src/clip_embeddings.py)).

![Method comparison](figures/ch05_method_comparison.png)

- **The venue trap.** Fed the full runway photo, CLIP identifies the collection **98%** of the time. It is recognizing the room (each show has its own set, lighting and crowd), not the clothes. Fed garment-only images (segmented, on a neutral background), it scores **42%**. All findings here use garment-only inputs. The full-photo embeddings also show a strong year trend (r = 0.54) that disappears once the venue is removed, so the apparent timeline came from changes in photography, not fashion.
- **Same accuracy, different eyes.** Garment-only CLIP and the color fingerprint both reach 42% but recognize different collections. Color identifies SS 2003 and SS 2014 best; CLIP identifies FW 2011 (87%) and FW 2001 (67%) best. Their collection-similarity rankings barely agree (Spearman ρ = 0.29). Together they reach 47%.
- **CLIP groups by silhouette and season.** Its closest pairs are SS 2005 & SS 2008 (summer dresses) and FW 2011 & FW 2016 (winter knit coats). The color fingerprint groups by palette instead.
- **Still no timeline.** Neither garment-only embedding correlates with year (|r| < 0.2).

<details><summary>CLIP versions of the chapter 04–05 charts</summary>

![CLIP similarity](figures/ch04_similarity_clip.png)
![CLIP era map](figures/ch05_era_map_clip.png)

</details>

### 06 — What makes it Missoni?

![Ablation](figures/ch06_ablation.png)

The demo below scores how close a garment is to the archive, calibrated so that **50 = a typical Missoni runway look** (each archive look scored with its own collection hidden). To find out what drives that score, 45 Missoni looks were re-scored with one ingredient removed at a time ([`src/ablation.py`](src/ablation.py)):

| condition | median score |
|---|---|
| original | 52 |
| no colour (greyscale) | 44 |
| light blur (pattern softened) | 23 |
| heavy blur (pattern gone, colour and drape kept) | 3 |
| one flat colour | 1 |

**To the model, Missoni-ness lives in the knit pattern, not the palette.** Removing all color costs little, but blurring away the pattern while keeping color, silhouette and drape erases the resemblance. Combined with chapter 05, this gives a two-part answer: **color tells Missoni collections apart, and pattern is what makes them Missoni.**

### Season effect: skin

![Skin share by season](figures/skin_share.png)

Spring/Summer looks show 27–33% skin, against 6–14% for Fall/Winter. **SS 2019 breaks the pattern at 11%**: it was as covered-up as a winter show, consistent with its muted palette.

## Demo: How Missoni Is This?

```bash
python app.py   # then open http://127.0.0.1:7860
```

Upload an outfit photo. The app segments the garment, embeds it with CLIP and returns a 0–100 score, the closest Missoni collection, the garment's palette and its nearest archive looks (with links to the archive lookbooks). There's also a command-line version: `python src/how_missoni.py photo.jpg`. *For fun and learning only. It cannot authenticate anything.*

## The investigations

| | chapter | question |
|---|---|---|
| 01 | **The Colors of Missoni** | How has the house palette changed by decade? |
| 02 | **Pattern DNA** | How often do zigzags, stripes and geometric patterns appear? |
| 03 | **Visual Complexity** | Are some eras more colorful or visually complex than others? |
| 04 | **Evolution vs. Consistency** | Which collections look most alike? |
| 05 | **Missoni's Eras** | Can unsupervised clustering find aesthetic periods without being told the year? |
| 06 | **What Makes It Missoni?** | Which visual ingredient (color, pattern or shape) carries the house identity? |

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

Every pixel is labelled as garment, skin, hair or background by **SegFormer-B2**, a transformer segmentation model fine-tuned for clothing ([mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes); NVIDIA SegFormer licence, non-commercial use). It runs locally with ONNX Runtime at about 0.7 s per look. CLIP ViT-B/32 (OpenAI, MIT licence) is run the same way, using the [Xenova ONNX export](https://huggingface.co/Xenova/clip-vit-base-patch32). Only garment pixels feed the color and pattern features, so the runway floor, background models and skin no longer distort the palette.

This replaced an earlier pipeline (fixed crop box plus a skin mask sampled from the model's face), which is kept as a fallback:

![Before/after: SS 2005 palette](figures/method_comparison_2005ss.png)

Pattern type is labelled per look (see [data/README.md](data/README.md#pattern-labels)) and used to train and test a classifier.

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
python src/import_lookbook.py pages data/lookbooks/<file>.pdf   # review a lookbook
python src/palette_strips.py      # -> figures/palette_strips.png, look_palettes.png
python src/chapter_charts.py      # -> figures/ch01–03, skin_share
python src/embeddings.py          # -> data/embeddings_handcrafted.csv
python src/era_analysis.py        # -> figures/ch04–05, data/era_results_handcrafted.json
python src/clip_embeddings.py     # -> data/embeddings_clip*.csv (first run downloads the 352 MB model)
python src/era_analysis.py clip   # CLIP versions; also: clip_fullframe, compare
python src/pattern_analysis.py    # -> figures/ch02b–c, data/pattern_results.json
python src/ablation.py            # -> figures/ch06_ablation.png
```

## Roadmap

- [x] Feature extraction pipeline
- [x] Collection palette strips
- [x] Adaptive skin masking
- [x] Garment segmentation (SegFormer-B2)
- [x] First collection: SS 2005, 15 looks
- [x] Nine collections spanning 2001–2023 (135 looks), imported automatically from the archive lookbooks
- [x] Pattern labels (AI-assisted, reviewed by the author) and pattern classifier (chapter 02b–c)
- [ ] Pre-2001 collections (Ottavio & Rosita era) from other sources
- [x] Chapters 01–03: palette, pattern and complexity over time
- [x] Hand-built visual fingerprints, collection similarity and era map (chapters 04–05)
- [x] CLIP image embeddings (garment-only), compared against the hand-built fingerprints
- [x] Ablation test: what drives the Missoni score (chapter 06)
- [x] *How Missoni Is This?* demo (Gradio app + CLI)
- [ ] Host the demo on Hugging Face Spaces

## Limitations

- The segmentation model was trained on everyday clothing, not runway knitwear. It treats swimwear as `top`, a jacket over a dress mostly as `dress`, and sheer fabric over skin as garment.
- Lookbook photography changes across the archive: scanned film in 2001–2003, then digital, with a different venue and lighting each season. Some cross-season color differences come from the camera rather than the clothes.
- The online archive starts at 2001, so the Ottavio & Rosita era is not yet covered. The SS 2019 show was co-ed and its lookbook includes menswear looks.
- A generic skin-color rule was tried for the fallback and rejected: it erased the beige, peach and yellow fabrics common in Missoni collections.
- Photography changes over 70 years (film stock, lighting, studio and runway shots), and this affects measured color. Treat cross-decade comparisons with care.
- Archive coverage is uneven, and early decades will have fewer looks.

---

*Part of a portfolio on data × culture × creativity — [madeleinelacoste](https://github.com/madeleinelacoste)*
