
"""Restore original Flickr30k captions without redownloading images."""

import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "flickr30k"
SPLITS = ("train", "validation", "test")


def normalize_captions(value):
    """Return non-empty captions from a string or sequence."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return []

    return [
        caption.strip()
        for caption in values
        if isinstance(caption, str) and caption.strip()
    ]


def main():
    print("Loading Flickr30k annotation metadata from cache...")
    dataset = load_dataset("AnyModal/flickr30k")

    for split in SPLITS:
        if split not in dataset:
            raise RuntimeError(
                f"Missing split {split}. "
                f"Available splits: {list(dataset.keys())}"
            )

        images_dir = DATA_ROOT / split / "images"
        output_csv = DATA_ROOT / split / "captions.csv"

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Existing image directory not found: {images_dir}"
            )

        # Exclude the image column so we only read caption metadata.
        metadata = dataset[split].select_columns(
            ["filename", "original_alt_text"]
        )

        rows = []
        caption_counts = Counter()
        missing_images = []
        invalid_annotations = []

        for index, record in enumerate(metadata):
            raw_filename = record.get("filename")
            if raw_filename:
                filename = Path(raw_filename).name
            else:
                filename = f"image_{index:05d}.jpg"

            if not (images_dir / filename).is_file():
                missing_images.append(filename)
                continue

            captions = normalize_captions(
                record.get("original_alt_text")
            )

            if len(captions) != 5:
                invalid_annotations.append(
                    (filename, len(captions))
                )
                continue

            caption_counts[filename] = len(captions)

            for caption_index, caption in enumerate(captions, start=1):
                rows.append({
                    "file_name": filename,
                    "caption": caption,
                    "caption_index": caption_index,
                })

        print(f"\n{split.upper()}")
        print(f"Images with five captions: {len(caption_counts)}")
        print(f"Caption rows prepared: {len(rows)}")
        print(f"Missing image files: {len(missing_images)}")
        print(
            "Images with other caption counts:",
            len(invalid_annotations),
        )

        if missing_images:
            print("Missing examples:", missing_images[:5])

        if invalid_annotations:
            print("Annotation examples:", invalid_annotations[:5])

        # Do not overwrite a split's CSV unless it validates completely.
        if (
            missing_images
            or invalid_annotations
            or len(caption_counts) != len(metadata)
        ):
            raise RuntimeError(
                f"{split}: annotations did not pass validation. "
                "The existing CSV has been left unchanged."
            )

        with output_csv.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "file_name",
                    "caption",
                    "caption_index",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Repaired annotations: {output_csv}")

    print("\nAll Flickr30k annotation splits repaired successfully.")


if __name__ == "__main__":
    main()
