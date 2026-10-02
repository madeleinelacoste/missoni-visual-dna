# Data

| file | who writes it | what it holds |
|---|---|---|
| `looks.csv` | **you, by hand** | one row per runway look: metadata + where the image came from |
| `images/` | you (local only, git-ignored) | the downloaded photos, named `{year}_{season}_{look:03d}.jpg` |
| `missoni_dataset.csv` | `src/extract_features.py` | `looks.csv` + every computed visual feature |

## `looks.csv` columns

| column | example | notes |
|---|---|---|
| `look_id` | `2021_FW_014` | `{year}_{season}_{look_number:03d}` |
| `year` | `2021` | |
| `season` | `FW` | `SS`, `FW`, `RESORT`, `PF` |
| `look_number` | `14` | position in the show/lookbook |
| `designer_era` | `Angela Missoni` | creative director at the time |
| `source` | `Vogue Runway` | `Archivio Missoni`, `Vogue Runway`, … |
| `source_url` | `https://…` | page the image came from — this is what makes the dataset reproducible |
| `image_path` | `images/2021_FW_014.jpg` | relative to `data/` |
| `pattern` | `zigzag` | the look's **most prominent** pattern: `zigzag`, `stripe`, `space-dye`, `geometric`, `print`, `plain` (see below) |
| `silhouette` | | optional hand label; the pipeline also writes `silhouette_auto` to `missoni_dataset.csv` |
| `notes` | | anything worth remembering |
| `pattern_reviewed` | `True` | the pattern label has been checked by a person (via `review_labels.py`) |

## Pattern labels

| label | covers |
|---|---|
| `zigzag` | zigzag, chevron and flame-stitch knits |
| `stripe` | horizontal or vertical stripes, including fringe stripes |
| `space-dye` | blurred multicolour yarns, flame-dye, ombré / dégradé |
| `geometric` | checks, plaids, argyle, patchwork, other geometric motifs |
| `print` | floral, figurative and graphic prints |
| `plain` | solid colour or tonal/textured knit without a distinct pattern |

**How they were made.** The labels are an AI-assisted first pass: Claude (an AI assistant) reviewed segmented garment close-ups of all 135 looks and assigned each look's most prominent pattern. Ambiguous calls (14 looks) are flagged with `pattern label uncertain` in `notes`. Labels checked by the project author are marked `pattern_reviewed = True`. Unreviewed labels are still the AI first pass. Run `python review_labels.py` to review them.

## Copyright

Runway and archive photographs belong to their photographers and publishers. The images stay on your machine (`data/images/` is git-ignored); the repo publishes only **source URLs and numbers derived from the images** (colors, scores, embeddings) plus the charts made from them.
