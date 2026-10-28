
"""Verify Flickr30k images and five-caption annotations."""

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "flickr30k"

EXPECTED_IMAGES = {
    "train": 29000,
    "validation": 1014,
    "test": 1000,
}


def verify_split(split, expected_count):
    split_dir = DATA_DIR / split
    image_dir = split_dir / "images"
    csv_path = split_dir / "captions.csv"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    image_files = {
        p.name for p in image_dir.iterdir()
        if p.is_file()
    }

    counts = Counter()
    empty_captions = 0
    unmatched_rows = 0

    with csv_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"file_name", "caption"}
        actual_columns = set(reader.fieldnames or [])

        if not required_columns.issubset(actual_columns):
            raise ValueError(
                f"{split}: expected columns {required_columns}, "
                f"found {actual_columns}"
            )

        for row in reader:
            filename = (row.get("file_name") or "").strip()
            caption = (row.get("caption") or "").strip()

            if not filename or filename not in image_files:
                unmatched_rows += 1
                continue

            if not caption:
                empty_captions += 1
                continue

            counts[filename] += 1

    wrong_caption_counts = {
        name: count
        for name, count in counts.items()
        if count != 5
    }

    images_without_captions = image_files - set(counts)

    checks = {
        "expected image count": len(image_files) == expected_count,
        "every image has captions": not images_without_captions,
        "exactly five captions per image": not wrong_caption_counts,
        "no unmatched rows": unmatched_rows == 0,
        "no empty captions": empty_captions == 0,
    }

    print(f"\n{split.upper()}")
    print(f"Images: {len(image_files)} / expected {expected_count}")
    print(f"Caption rows: {sum(counts.values())}")
    print(f"Images with captions: {len(counts)}")
    print(f"Images with incorrect caption counts: {len(wrong_caption_counts)}")
    print(f"Images without captions: {len(images_without_captions)}")
    print(f"Unmatched rows: {unmatched_rows}")
    print(f"Empty captions: {empty_captions}")

    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {label}")

    return all(checks.values())


def main():
    all_passed = True

    for split, expected in EXPECTED_IMAGES.items():
        try:
            if not verify_split(split, expected):
                all_passed = False
        except (OSError, ValueError, csv.Error) as exc:
            print(f"\nERROR in {split}: {exc}")
            all_passed = False

    print("\n" + "=" * 45)
    print("RESULT:", "ALL CHECKS PASSED" if all_passed else "CHECK FAILED")
    print("=" * 45)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
