import os
from preprocess import clean_caption
IMAGE_DIR = "data/images"
CAPTION_FILE = "data/captions/captions.txt"
print("Image Folder Exists:", os.path.exists(IMAGE_DIR))
print("Caption File Exists:", os.path.exists(CAPTION_FILE))
images = os.listdir(IMAGE_DIR)


def load_captions():
    with open(CAPTION_FILE, "r", encoding="utf-8") as file:
        lines = file.readlines()



    captions_dict={}
    for line in lines[1:]:
        line = line.strip()
        image, caption = line.split(',',1)
    
        if image not in captions_dict:
            captions_dict[image]=[]
        
        captions_dict[image].append(caption)
    return captions_dict


# cleaning all
def clean_all_captions(captions_dict):
    for image_name in captions_dict:
        captions_dict[image_name]=[
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

    # Dataset
#         │
#         ▼
# captions.txt
#         │
#         ▼
# Read Lines
#         │
#         ▼
# Split Image + Caption
#         │
#         ▼
# Dictionary