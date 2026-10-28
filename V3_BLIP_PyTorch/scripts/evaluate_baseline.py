
import csv
import time
from datetime import datetime
from pathlib import Path

from src.captioner import ImageCaptioner

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results" / "baseline"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    # Load model only once.
    captioner = ImageCaptioner()
    rows = []

    for i, image_path in enumerate(images, start=1):
        start = time.perf_counter()

        try:
            caption = captioner.generate_caption(image_path)
            elapsed = time.perf_counter() - start

            print(f"\n[{i}/{len(images)}] {image_path.name}")
            print(f"Caption: {caption}")
            print(f"Time: {elapsed:.3f} seconds")

            rows.append({
                "image": image_path.name,
                "caption": caption,
                "inference_seconds": round(elapsed, 4),
                "status": "success",
            })

        except Exception as exc:
            print(f"Failed on {image_path.name}: {exc}")
            rows.append({
                "image": image_path.name,
                "caption": "",
                "inference_seconds": "",
                "status": f"failed: {exc}",
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("captions_%Y%m%d_%H%M%S.csv")
    output_file = OUTPUT_DIR / filename

    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "caption",
                "inference_seconds",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nProcessed {len(rows)} images.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
