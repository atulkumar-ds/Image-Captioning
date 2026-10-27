import os
import sys
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
V1_ROOT = SRC_DIR.parent
REPO_ROOT = V1_ROOT.parent

# Set working directory to V1_ROOT so legacy dataset.py relative paths resolve
os.chdir(V1_ROOT)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import load_captions, clean_all_captions
from generate_caption import load_tokenizer, load_caption_model, generate_caption
from cnn_features import get_cnn_model, extract_feature

DEFAULT_MODEL_FILE = V1_ROOT / "models" / "best_caption_model.keras"
DEFAULT_TOKENIZER_FILE = V1_ROOT / "models" / "tokenizer.json"
DEFAULT_IMAGE_DIR = V1_ROOT / "data" / "images"


def evaluate_sample(
    image_name=None,
    model=None,
    cnn=None,
    word_to_index=None,
    index_to_word=None,
    max_length=None,
    model_file=None,
    tokenizer_file=None,
    image_dir=None
):
    """
    Evaluates and displays captions for a single image sample.
    """
    if tokenizer_file is None:
        tokenizer_file = DEFAULT_TOKENIZER_FILE
    if model_file is None:
        model_file = DEFAULT_MODEL_FILE
    if image_dir is None:
        image_dir = DEFAULT_IMAGE_DIR

    if word_to_index is None or index_to_word is None or max_length is None:
        word_to_index, index_to_word, max_length = load_tokenizer(tokenizer_file)

    if model is None:
        model = load_caption_model(
            model_file=model_file,
            vocab_size=len(word_to_index),
            max_length=max_length
        )

    if cnn is None:
        cnn = get_cnn_model()

    captions_dict = clean_all_captions(load_captions())

    if image_name is None:
        image_name = "416650559_cd08d3cd96.jpg"

    image_path = Path(image_dir) / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\n" + "=" * 80)
    print(f"IMAGE SAMPLE EVALUATION: {image_name}")
    print("=" * 80)

    print("\nExtracting CNN feature...")
    image_feature = extract_feature(str(image_path), cnn)

    print("Generating caption...")
    predicted_caption = generate_caption(
        model=model,
        image_feature=image_feature,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        max_length=max_length
    )

    if image_name in captions_dict:
        print("\nGROUND TRUTH CAPTIONS:")
        for ref in captions_dict[image_name]:
            print(" -", ref)

    print("\nPREDICTED CAPTION:")
    print(predicted_caption)
    print("=" * 80 + "\n")

    return predicted_caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate single image caption using V1 Baseline.")
    parser.add_argument("--image", type=str, default="416650559_cd08d3cd96.jpg", help="Image filename")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_FILE), help="Model file path")
    parser.add_argument("--tokenizer", type=str, default=str(DEFAULT_TOKENIZER_FILE), help="Tokenizer path")
    args = parser.parse_args()

    evaluate_sample(
        image_name=args.image,
        model_file=args.model,
        tokenizer_file=args.tokenizer
    )
