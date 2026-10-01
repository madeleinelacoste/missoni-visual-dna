"""How Missoni Is This? A small web demo.

    python app.py   # then open http://127.0.0.1:7860

Upload an outfit photo; the garment is segmented, embedded with CLIP and
compared with 135 Missoni runway looks (2001–2023). Not an authentication
tool, just a demonstration of image embeddings and similarity.

Archive photos are shown only when they exist locally (data/images/ is never
published); otherwise matches are listed with links to the archive.
"""

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from how_missoni import analyse, verdict  # noqa: E402

ABOUT = """
**How it works.** The model isolates the garment, turns it into a CLIP embedding and finds the closest
looks among 135 Missoni runway looks from 2001–2023. The score is calibrated against Missoni itself:
**50 means as Missoni as a typical Missoni runway look**.

What drives the score? An [ablation test](https://github.com/madeleinelacoste/missoni-visual-dna#06--what-makes-it-missoni)
shows it is mostly the **knit pattern**: blurring the pattern away drops a Missoni look from 52 to 3, while removing
all colour only drops it to 44.

*For fun and for learning. This cannot tell you whether something is authentic.*
"""


def palette_html(palette) -> str:
    cells = "".join(
        f'<div title="{h}" style="flex:{share:.3f};background:{h};height:36px"></div>' for h, share in palette
    )
    return f'<div style="display:flex;gap:2px;border-radius:4px;overflow:hidden">{cells}</div>'


def run(path):
    if path is None:
        return "Upload a photo to begin.", None, "", [], ""
    r = analyse(path)
    if not r.garment_found:
        return "### No garment detected\nTry a photo where one outfit is clearly visible.", r.garment, "", [], ""
    headline = (
        f"# {r.score:.0f} / 100\n### {verdict(r.score)}\n"
        f"As Missoni as **{r.score:.0f}%** of Missoni's own runway looks.  \n"
        f"Closest collection: **{r.closest_collection}**"
    )
    gallery, rows = [], []
    for nb in r.neighbours.itertuples():
        img = ROOT / "data" / nb.image_path
        if img.exists():
            gallery.append((str(img), f"{nb.collection} · {nb.similarity:.2f}"))
        rows.append(f"| {nb.collection} | {nb.similarity:.3f} | [archive lookbook]({nb.source_url}) |")
    table = "| Collection | Similarity | Source |\n|---|---|---|\n" + "\n".join(rows)
    return headline, r.garment, palette_html(r.palette), gallery, table


with gr.Blocks(title="How Missoni Is This?") as demo:
    gr.Markdown("# How Missoni Is This?\nUpload an outfit photo and see how close it is to the Missoni runway archive.")
    with gr.Row():
        with gr.Column(scale=1):
            photo = gr.Image(type="filepath", label="Outfit photo", height=420)
            go = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=1):
            headline = gr.Markdown()
            garment = gr.Image(label="What the model looked at (garment only)", height=260, interactive=False)
            palette = gr.HTML()
    gr.Markdown("### Closest Missoni looks")
    gallery = gr.Gallery(columns=5, height=300, show_label=False)
    matches = gr.Markdown()
    gr.Markdown(ABOUT)
    go.click(run, inputs=photo, outputs=[headline, garment, palette, gallery, matches])
    photo.upload(run, inputs=photo, outputs=[headline, garment, palette, gallery, matches])

if __name__ == "__main__":
    demo.launch()
