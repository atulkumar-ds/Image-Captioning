
"""Compare pretrained BLIP with the Flickr30k fine-tuned checkpoint."""

import argparse
import csv
import gc
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "flickr30k" / "test"
IMAGE_DIR = TEST_DIR / "images"
CAPTIONS_CSV = TEST_DIR / "captions.csv"

BASELINE_MODEL = "Salesforce/blip-image-captioning-base"
FINETUNED_MODEL = ROOT / "checkpoints" / "flickr30k_blip"
OUTPUT_DIR = ROOT / "results" / "evaluation"


def load_test_data(max_images: int, seed: int):
    """Load reference captions and select a reproducible image subset."""
    references = {}

    with CAPTIONS_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {"file_name", "caption"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Unexpected caption CSV columns: {reader.fieldnames}"
            )

        for row in reader:
            filename = Path(row["file_name"]).name
            caption = (row["caption"] or "").strip()

            if caption:
                references.setdefault(filename, []).append(caption)

    # Only include images that exist and have exactly five references.
    valid_filenames = sorted(
        filename
        for filename, captions in references.items()
        if len(captions) == 5
        and (IMAGE_DIR / filename).is_file()
    )

    if not valid_filenames:
        raise RuntimeError(
            "No usable test images found. Check the Flickr30k test split."
        )

    if max_images > 0 and max_images < len(valid_filenames):
        rng = random.Random(seed)
        valid_filenames = sorted(
            rng.sample(valid_filenames, max_images)
        )

    return valid_filenames, references


def get_autocast_context(device):
    """Use supported mixed precision for inference when on CUDA."""
    if device.type != "cuda":
        return nullcontext()

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    return torch.autocast(device_type="cuda", dtype=dtype)


def generate_predictions(
    model_name,
    label,
    filenames,
    references,
    device,
    output_file,
    max_new_tokens,
):
    """Generate and save captions for one model."""
    print(f"\nLoading {label}: {model_name}")

    processor = BlipProcessor.from_pretrained(
        str(model_name),
        use_fast=False,
    )
    model = BlipForConditionalGeneration.from_pretrained(
        str(model_name)
    ).to(device)
    model.eval()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image",
        "caption",
        "references",
        "inference_seconds",
        "model",
    ]

    total_seconds = 0.0

    # Write incrementally so completed predictions survive interruptions.
    with output_file.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        file.flush()

        for index, filename in enumerate(filenames, start=1):
            image_path = IMAGE_DIR / filename

            with Image.open(image_path) as image:
                image = image.convert("RGB")

                inputs = processor(
                    images=image,
                    return_tensors="pt",
                ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            started = time.perf_counter()

            with torch.inference_mode():
                with get_autocast_context(device):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=3,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - started
            total_seconds += elapsed

            caption = processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()

            writer.writerow({
                "image": filename,
                "caption": caption,
                "references": json.dumps(
                    references[filename], ensure_ascii=False
                ),
                "inference_seconds": round(elapsed, 4),
                "model": label,
            })
            file.flush()

            if index == 1 or index % 25 == 0 or index == len(filenames):
                print(
                    f"{label}: {index}/{len(filenames)} | "
                    f"Latest caption: {caption}"
                )

    average_seconds = total_seconds / len(filenames)

    print(f"Saved predictions to: {output_file}")
    print(f"Average generation time: {average_seconds:.3f} s/image")

    del model, processor
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Compare pretrained and fine-tuned BLIP on Flickr30k."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Images to evaluate; use 0 for all test images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    if args.max_images < 0:
        parser.error("--max-images must be zero or positive.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    if not (FINETUNED_MODEL / "config.json").is_file():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {FINETUNED_MODEL}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    filenames, references = load_test_data(
        args.max_images,
        args.seed,
    )

    print(f"Device: {device}")
    print(f"Test images selected: {len(filenames)}")
    print("Reference captions per image: 5")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_predictions(
        model_name=BASELINE_MODEL,
        label="pretrained_blip",
        filenames=filenames,
        references=references,
        device=device,
        output_file=OUTPUT_DIR / "test_predictions_baseline.csv",
        max_new_tokens=args.max_new_tokens,
    )

    generate_predictions(
        model_name=FINETUNED_MODEL,
        label="flickr30k_finetuned_blip",
        filenames=filenames,
        references=references,
        device=device,
        output_file=OUTPUT_DIR / "test_predictions_finetuned.csv",
        max_new_tokens=args.max_new_tokens,
    )

    print("\nCaption generation comparison complete.")
    print(f"Results directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
