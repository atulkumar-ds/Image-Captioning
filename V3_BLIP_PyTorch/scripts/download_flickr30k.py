
import csv
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "flickr30k"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Flickr30k from Hugging Face...")
    dataset = load_dataset("AnyModal/flickr30k")

    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            raise RuntimeError(
                f"Missing expected split: {split_name}. "
                f"Available splits: {list(dataset.keys())}"
            )

        split_dir = OUTPUT_DIR / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        annotation_path = split_dir / "captions.csv"
        saved = 0

        with annotation_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["file_name", "caption"])

            for index, row in enumerate(dataset[split_name]):
                filename = row.get("filename")

                if not filename:
                    filename = f"image_{index:05d}.jpg"
                else:
                    filename = Path(filename).name

                image_path = images_dir / filename

                # Save the image only if it isn't already present.
                if not image_path.exists():
                    row["image"].convert("RGB").save(
                        image_path, format="JPEG", quality=95
                    )

                captions = row["alt_text"]
                if isinstance(captions, str):
                    captions = [captions]

                for caption in captions:
                    if caption and caption.strip():
                        writer.writerow([filename, caption.strip()])

                saved += 1

                if saved % 500 == 0:
                    print(
                        f"{split_name}: "
                        f"{saved}/{len(dataset[split_name])} images"
                    )

        print(f"{split_name}: saved {saved} images")
        print(f"Annotations: {annotation_path}")

    print("\nFlickr30k preparation completed.")
    print(f"Dataset directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
