
"""Print a side-by-side qualitative caption comparison."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "evaluation"
BASELINE = RESULTS_DIR / "test_predictions_baseline.csv"
FINETUNED = RESULTS_DIR / "test_predictions_finetuned.csv"


def load_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return {
            row["image"]: row
            for row in csv.DictReader(file)
        }


def main():
    baseline = load_csv(BASELINE)
    finetuned = load_csv(FINETUNED)

    if set(baseline) != set(finetuned):
        raise ValueError("The two prediction files do not match.")

    print(f"Comparing {len(baseline)} test images.\n")

    # Print 30 examples to make manual review manageable.
    for index, filename in enumerate(sorted(baseline)[:30], start=1):
        b = baseline[filename]
        f = finetuned[filename]

        references = json.loads(b["references"])

        print("=" * 80)
        print(f"{index}. IMAGE: {filename}")
        print(f"ORIGINAL BLIP: {b['caption']}")
        print(f"FINE-TUNED BLIP: {f['caption']}")
        print("REFERENCE CAPTIONS:")

        for ref_index, reference in enumerate(references, start=1):
            print(f"  {ref_index}. {reference}")

    print("=" * 80)
    print("Review these examples for accuracy, detail and hallucinations.")


if __name__ == "__main__":
    main()
