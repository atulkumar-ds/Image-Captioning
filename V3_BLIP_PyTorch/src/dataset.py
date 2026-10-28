
"""PyTorch dataset and collator for Flickr30k captions."""

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class Flickr30kCaptionDataset(Dataset):
    """One dataset item represents one image-caption pair."""

    def __init__(
        self,
        split_dir: str | Path,
        max_rows: int = 0,
        seed: int = 42,
    ):
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "images"
        self.csv_path = self.split_dir / "captions.csv"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Caption CSV not found: {self.csv_path}"
            )

        with self.csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)
            required = {"file_name", "caption"}

            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"Expected columns {required}, "
                    f"found {reader.fieldnames}"
                )

            rows = []
            for row in reader:
                filename = Path(row["file_name"]).name
                caption = (row["caption"] or "").strip()
                image_path = self.image_dir / filename

                if not caption:
                    continue

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image referenced in CSV is missing: {image_path}"
                    )

                rows.append({
                    "image_path": image_path,
                    "caption": caption,
                })

        if not rows:
            raise ValueError(f"No usable caption rows in {self.csv_path}")

        # A reproducible subset is useful for short smoke tests.
        if max_rows > 0 and len(rows) > max_rows:
            rng = random.Random(seed)
            rows = rng.sample(rows, max_rows)

        self.rows = rows
        print(
            f"Loaded {len(self.rows):,} caption pairs "
            f"from {self.split_dir.name}"
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")

        return image, row["caption"]


class BLIPCaptionCollator:
    """Convert images and captions into BLIP tensors."""

    def __init__(self, processor, max_length: int = 40):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        images, captions = zip(*batch)

        encoded = self.processor(
            images=list(images),
            text=list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = encoded["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id

        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {
            "pixel_values": encoded["pixel_values"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
