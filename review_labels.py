"""Review the pattern labels in data/looks.csv, one look at a time.

    python review_labels.py   # then open http://127.0.0.1:7861

Click the right pattern (or Keep) and the page moves to the next look. Every
answer is saved to looks.csv straight away, with pattern_reviewed = True, so
you can stop and come back later. Looks flagged "pattern label uncertain"
come first. When you're done, "Rebuild charts" re-runs the pattern analysis.
"""

import subprocess
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOOKS = ROOT / "data" / "looks.csv"
CROPS = ROOT / "data" / "crops"
PATTERNS = ["zigzag", "stripe", "space-dye", "geometric", "print", "plain"]
DEFINITIONS = (
    "**zigzag** zigzag, chevron, flame-stitch · **stripe** stripes incl. fringe · "
    "**space-dye** blurred multicolour yarn, ombré · **geometric** checks, plaid, argyle, patchwork · "
    "**print** floral, figurative, graphic · **plain** solid or tonal texture"
)


def load() -> pd.DataFrame:
    df = pd.read_csv(LOOKS)
    if "pattern_reviewed" not in df:
        df["pattern_reviewed"] = False
    df["pattern_reviewed"] = df["pattern_reviewed"].fillna(False).astype(bool)
    return df


def order(df: pd.DataFrame, only_unreviewed: bool) -> list[str]:
    """Uncertain looks first, then the rest in collection order."""
    uncertain = df["notes"].fillna("").str.contains("uncertain")
    ranked = df.assign(_u=~uncertain).sort_values(["_u", "year", "season", "look_number"])
    if only_unreviewed:
        ranked = ranked[~ranked["pattern_reviewed"]]
    return list(ranked["look_id"])


def view(queue: list[str], i: int):
    df = load()
    done = int(df["pattern_reviewed"].sum())
    progress = f"**{done} / {len(df)} reviewed**"
    if not queue or i >= len(queue):
        return None, None, f"### All done in this list\n{progress}", gr.update(visible=True)
    row = df.set_index("look_id").loc[queue[i]]
    flag = " · ⚠️ flagged uncertain" if "uncertain" in str(row["notes"]) else ""
    status = " · ✓ reviewed" if row["pattern_reviewed"] else ""
    info = (
        f"### {queue[i]}{flag}{status}\n"
        f"Current label: **{row['pattern']}**  \n"
        f"Look {i + 1} of {len(queue)} in this list · {progress}  \n"
        f"[Source page]({row['source_url']})"
    )
    return str(ROOT / "data" / row["image_path"]), str(CROPS / f"{queue[i]}.jpg"), info, gr.update(visible=done > 0)


def answer(label: str | None, queue: list[str], i: int):
    if queue and i < len(queue):
        df = load()
        m = df["look_id"] == queue[i]
        if label:
            df.loc[m, "pattern"] = label
        df.loc[m, "pattern_reviewed"] = True
        df.to_csv(LOOKS, index=False)
    return i + 1, *view(queue, i + 1)


def back(queue: list[str], i: int):
    i = max(0, i - 1)
    return i, *view(queue, i)


def restart(only_unreviewed: bool):
    queue = order(load(), only_unreviewed)
    return queue, 0, *view(queue, 0)


def rebuild():
    out = subprocess.run([sys.executable, "src/pattern_analysis.py"], cwd=ROOT, capture_output=True, text=True)
    return "Charts rebuilt." if out.returncode == 0 else f"Error:\n```\n{out.stderr[-1500:]}\n```"


with gr.Blocks(title="Review pattern labels") as demo:
    gr.Markdown("# Review pattern labels\nPick the look's **most prominent** pattern. " + DEFINITIONS)
    queue, idx = gr.State([]), gr.State(0)
    with gr.Row():
        full = gr.Image(label="Runway look", height=560, interactive=False)
        crop = gr.Image(label="Garment close-up", height=560, interactive=False)
        with gr.Column():
            info = gr.Markdown()
            buttons = [gr.Button(p, variant="secondary") for p in PATTERNS]
            keep = gr.Button("✓ Keep current label", variant="primary")
            with gr.Row():
                prev = gr.Button("← Back")
                only = gr.Checkbox(label="Only unreviewed", value=True)
            rebuild_btn = gr.Button("Rebuild charts with my labels", visible=False)
            rebuilt = gr.Markdown()
    outputs = [idx, full, crop, info, rebuild_btn]
    for b, p in zip(buttons, PATTERNS):
        b.click(lambda q, i, p=p: answer(p, q, i), [queue, idx], outputs)
    keep.click(lambda q, i: answer(None, q, i), [queue, idx], outputs)
    prev.click(back, [queue, idx], outputs)
    only.change(restart, only, [queue, *outputs])
    rebuild_btn.click(rebuild, None, rebuilt)
    demo.load(restart, only, [queue, *outputs])

if __name__ == "__main__":
    demo.launch(server_port=7861)
