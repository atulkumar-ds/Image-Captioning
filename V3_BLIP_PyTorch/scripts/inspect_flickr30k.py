
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "flickr30k"

for split in ("train", "validation", "test"):
    path = DATA_DIR / split / "captions.csv"

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        counts = Counter()
        total_rows = 0

        for row in reader:
            counts[row["file_name"]] += 1
            total_rows += 1

    distribution = Counter(counts.values())

    print(f"\n{split.upper()}")
    print("CSV caption rows:", total_rows)
    print("Unique images with captions:", len(counts))
    print("Captions per image distribution:", dict(sorted(distribution.items())))
    print("Example counts:", list(counts.items())[:5])
