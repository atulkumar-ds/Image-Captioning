import string
import re

VALID_SINGLE_LETTERS = {"a", "i"}

def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(
        caption.maketrans('', '', string.punctuation)
    )

    caption = re.sub(r"\d+", "", caption)

    caption = " ".join(caption.split())
    words = caption.split()

    # Preserve valid single-letter English words ("a", "i") while removing isolated noise/artifacts
    words = [word for word in words if len(word) > 1 or word in VALID_SINGLE_LETTERS]

    caption = " ".join(words)
    caption = "<start> " + caption + " <end>"
    
    return caption
