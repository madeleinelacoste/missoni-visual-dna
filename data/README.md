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
| `pattern` | `zigzag` | **optional, hand-labelled on a subset**: `zigzag`, `stripe`, `geometric`, `space-dye`, `floral-abstract`, `solid`, `other` |
| `silhouette` | `dress` | optional: `dress`, `top+skirt`, `top+trousers`, `coat`, `knit-set`, `other` |
| `notes` | | anything worth remembering |

## Copyright

Runway and archive photographs belong to their photographers and publishers. The images stay on your machine (`data/images/` is git-ignored); the repo publishes only **source URLs and numbers derived from the images** (colors, scores, embeddings) plus the charts made from them.
