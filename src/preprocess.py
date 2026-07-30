import string
import re

def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(
        caption.maketrans('', '', string.punctuation)
    )

    caption = re.sub(r"\d+","", caption)

    caption = " ".join(caption.split())
    words =  caption.split()

    words = [word for word in words if len(word)>1]

    caption = " ".join(words)
    caption = "<start> " + caption + " <end>"
    
    return caption

