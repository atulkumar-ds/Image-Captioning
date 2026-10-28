
"""Prepare reproducible MS COCO 2017 image-caption manifests.

Expected dataset structure:
data/
    coco2017/
        train2017/
        val2017/
        annotations/
            captions_train2017.json
            captions_val2017.json

This script does not download images or train the model.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COCO_ROOT = ROOT / "data" / "coco2017"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "manifests"


def load_annotations(annotation_file):
    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {
        image["id"]: image
        for image in data.get("images", [])
    }
    annotations = data.get("annotations", [])

    if not images or not annotations:
        raise ValueError(
            f"No usable images or captions found in {annotation_file}"
        )

    return images, annotations


def prepare_split(
    split,
    image_dir,
    annotation_file,
    max_images,
    seed,
):
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    images, annotations = load_annotations(annotation_file)

    # Group captions by image ID.
    captions_by_image = {}
    for annotation in annotations:
        image_id = annotation.get("image_id")
        caption = annotation.get("caption", "").strip()

        if image_id in images and caption:
            captions_by_image.setdefault(image_id, []).append(caption)

    # Keep only images that have captions.
    eligible_ids = sorted(
        image_id
        for image_id in captions_by_image
        if images[image_id].get("file_name")
    )

    if not eligible_ids:
        raise ValueError(f"No image-caption pairs found for {split}")

    # Deterministic image-level subset selection.
    if max_images is not None and max_images > 0:
        rng = random.Random(seed)
        if len(eligible_ids) > max_images:
            eligible_ids = sorted(
                rng.sample(eligible_ids, max_images)
            )

    rows = []
    missing_images = []

    for image_id in eligible_ids:
        filename = images[image_id]["file_name"]
        image_path = image_dir / filename

        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue

        for caption in captions_by_image[image_id]:
            rows.append({
                "split": split,
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "file_name": filename,
                "caption": caption,
            })

    return rows, len(eligible_ids), missing_images


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "image_id",
        "image_path",
        "file_name",
        "caption",
    ]

    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MS COCO image-caption CSV manifests."
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=DEFAULT_COCO_ROOT,
        help="Root containing train2017, val2017 and annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=5000,
        help="Maximum training images; use 0 for all images.",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=1000,
        help="Maximum validation images; use 0 for all images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_train_images < 0 or args.max_val_images < 0:
        parser.error("Image limits must be zero or positive.")

    root = args.coco_root
    annotations_dir = root / "annotations"

    try:
        train_rows, train_count, train_missing = prepare_split(
            split="train",
            image_dir=root / "train2017",
            annotation_file=annotations_dir / "captions_train2017.json",
            max_images=args.max_train_images,
            seed=args.seed,
        )

        val_rows, val_count, val_missing = prepare_split(
            split="validation",
            image_dir=root / "val2017",
            annotation_file=annotations_dir / "captions_val2017.json",
            max_images=args.max_val_images,
            seed=args.seed + 1,
        )

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"\nDATASET PREPARATION FAILED\n{exc}")
        print("\nExpected structure:")
        print(root / "train2017")
        print(root / "val2017")
        print(annotations_dir / "captions_train2017.json")
        print(annotations_dir / "captions_val2017.json")
        sys.exit(1)

    all_missing = train_missing + val_missing

    if all_missing:
        print(f"\nMissing images: {len(all_missing)}")
        for path in all_missing[:20]:
            print(f"  {path}")
        if len(all_missing) > 20:
            print(f"  ... and {len(all_missing) - 20} more")

    if not train_rows or not val_rows:
        print("\nNo usable training or validation rows.")
        print("Check that the image files match the annotation JSON.")
        sys.exit(1)

    train_output = args.output_dir / "train.csv"
    val_output = args.output_dir / "validation.csv"

    write_manifest(train_output, train_rows)
    write_manifest(val_output, val_rows)

    print("\nDATASET PREPARATION COMPLETE")
    print("-" * 45)
    print(f"Training images selected:   {train_count}")
    print(f"Training caption pairs:     {len(train_rows)}")
    print(f"Validation images selected: {val_count}")
    print(f"Validation caption pairs:   {len(val_rows)}")
    print(f"Missing image files:        {len(all_missing)}")
    print(f"\nTraining manifest:   {train_output}")
    print(f"Validation manifest: {val_output}")
    print(f"Random seed: {args.seed}")


if __name__ == "__main__":
    main()
