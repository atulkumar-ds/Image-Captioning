
"""Standalone BLIP image captioning inference module."""

from pathlib import Path
import argparse

import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageCaptioner:
    """Load a pretrained BLIP model and generate image captions."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        print(f"Loading BLIP model: {model_name}")
        print(f"Device: {DEVICE}")

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name
        ).to(DEVICE)

        self.model.eval()
        print("BLIP model loaded successfully.")

    @torch.inference_mode()
    def generate_caption(
        self,
        image_path: str | Path,
        max_new_tokens: int = 30,
    ) -> str:
        """Generate a caption for a local image file."""
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path.resolve()}"
            )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Could not read image: {image_path}"
            ) from exc

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
        )

        caption = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        return caption


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a caption for an image using BLIP."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the image file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1.")

    try:
        captioner = ImageCaptioner()
        caption = captioner.generate_caption(
            args.image,
            max_new_tokens=args.max_new_tokens,
        )

        print("\n" + "=" * 55)
        print("GENERATED CAPTION")
        print("=" * 55)
        print(caption)
        print("=" * 55)

    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
