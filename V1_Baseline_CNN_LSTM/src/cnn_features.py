import os
from pathlib import Path
import numpy as np

from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

SRC_DIR = Path(__file__).resolve().parent
V1_ROOT = SRC_DIR.parent

IMAGE_DIR = str(V1_ROOT / "data" / "images")
FEATURE_FILE = str(V1_ROOT / "data" / "features" / "image_features.npz")

_cnn_model = None


def get_cnn_model():
    """Lazily instantiates the InceptionV3 feature extractor on first call."""
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )
        _cnn_model.trainable = False
    return _cnn_model


class _LazyCNNProxy:
    """Transparent proxy that defers model loading until an attribute is accessed or called."""
    def __getattr__(self, name):
        return getattr(get_cnn_model(), name)

    def __setattr__(self, name, value):
        setattr(get_cnn_model(), name, value)

    def __call__(self, *args, **kwargs):
        return get_cnn_model()(*args, **kwargs)

    def __repr__(self):
        return f"<LazyCNNProxy for {get_cnn_model()!r}>"

    def __bool__(self):
        return True


cnn_model = _LazyCNNProxy()


def extract_feature(image_path, model=None):
    if model is None:
        model = get_cnn_model()

    img = image.load_img(
        str(image_path),
        target_size=(299, 299)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Direct layer invocation with training=False avoids predict overhead
    features = model(img_array, training=False).numpy()

    return features[0]


def extract_all_features(image_dir=None, model=None):
    if image_dir is None:
        image_dir = IMAGE_DIR
    if model is None:
        model = get_cnn_model()

    features_dict = {}
    image_names = os.listdir(image_dir)
    image_names = [
        name for name in image_names
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total_images = len(image_names)
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(image_dir, image_name)
        try:
            features = extract_feature(image_path, model)
            features_dict[image_name] = features
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == total_images:
            print(f"Processed {i + 1}/{total_images}")

    return features_dict


def save_features(features_dict, file_path=None):
    if file_path is None:
        file_path = FEATURE_FILE

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez_compressed(file_path, **features_dict)
    print("\nFeatures saved successfully.")
    print("Location:", file_path)


def load_features(file_path=None):
    if file_path is None:
        file_path = FEATURE_FILE

    data = np.load(file_path, allow_pickle=True)
    if hasattr(data, "files"):
        features_dict = {
            image_name: data[image_name]
            for image_name in data.files
        }
    elif hasattr(data, "item"):
        features_dict = data.item()
    else:
        features_dict = dict(data)
    return features_dict


if __name__ == "__main__":
    print("Starting CNN feature extraction...\n")
    features_dict = extract_all_features(IMAGE_DIR, cnn_model)
    print("\nTotal Features Extracted:", len(features_dict))
    save_features(features_dict, FEATURE_FILE)
