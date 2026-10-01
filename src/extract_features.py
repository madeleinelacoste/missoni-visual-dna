"""looks.csv (hand-made metadata) -> missoni_dataset.csv (metadata + visual features).

    python src/extract_features.py
"""

from pathlib import Path

import pandas as pd

from visual_features import extract

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    looks = pd.read_csv(DATA / "looks.csv")
    rows, missing = [], []
    for _, look in looks.iterrows():
        path = DATA / look["image_path"]
        if not path.exists():
            missing.append(look["look_id"])
            continue
        rows.append({"look_id": look["look_id"], **extract(str(path))})
        print(f"  {look['look_id']}")

    dataset = looks.merge(pd.DataFrame(rows), on="look_id", how="inner")
    dataset.to_csv(DATA / "missoni_dataset.csv", index=False)
    print(f"\n{len(dataset)} looks -> data/missoni_dataset.csv")
    if missing:
        print(f"{len(missing)} images not found (skipped): {', '.join(missing)}")


if __name__ == "__main__":
    main()
