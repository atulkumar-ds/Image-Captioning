import os
from pathlib import Path
from preprocess import clean_caption

SRC_DIR = Path(__file__).resolve().parent
V1_ROOT = SRC_DIR.parent

IMAGE_DIR = str(V1_ROOT / "data" / "images")
CAPTION_FILE = str(V1_ROOT / "data" / "captions" / "captions.txt")
images = os.listdir(IMAGE_DIR) if os.path.exists(IMAGE_DIR) else []


def load_captions(caption_file=None):
    file_path = Path(caption_file).resolve() if caption_file else Path(CAPTION_FILE)
    if not file_path.exists():
        raise FileNotFoundError(f"Caption file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    captions_dict = {}
    for line in lines[1:]:
        line = line.strip()
        if not line or ',' not in line:
            continue
        image, caption = line.split(',', 1)

        if image not in captions_dict:
            captions_dict[image] = []

        captions_dict[image].append(caption)
    return captions_dict


# cleaning all
def clean_all_captions(captions_dict):
    for image_name in captions_dict:
        captions_dict[image_name] = [
            clean_caption(caption) for caption in captions_dict[image_name]
        ]

    return captions_dict


if __name__ == "__main__":
    captions_dict = load_captions()
    captions_dict = clean_all_captions(captions_dict)

    print("Total Images:", len(captions_dict))

    first_image = list(captions_dict.keys())[0]

    print("\nImage:", first_image)

    print("\nCaptions:")

    for caption in captions_dict[first_image]:
        print(caption)
