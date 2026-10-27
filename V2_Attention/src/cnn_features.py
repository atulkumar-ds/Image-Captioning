import os
import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
V2_ROOT = SRC_DIR.parent
REPO_ROOT = V2_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from V2_Attention.src.feature_store import MmapFeatureStore
except ImportError:
    from feature_store import MmapFeatureStore

from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

IMAGE_DIR = str(REPO_ROOT / "V1_Baseline_CNN_LSTM" / "data" / "images")
FEATURE_FILE = str(V2_ROOT / "model" / "v2_features.npy")

_cnn_model = None


def get_cnn_model():
    """Lazily instantiates the InceptionV3 feature extractor on first call."""
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = InceptionV3(
            weights="imagenet",
            include_top=False
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

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Direct model invocation avoids repeated predict overhead
    features = model(img, training=False).numpy()

    return features


def load_features(feature_file=None):
    if feature_file is None:
        feature_file = FEATURE_FILE

    path = Path(feature_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Feature file or directory not found at: {path}")

    # Memory-mapped feature store (manifest JSON or directory containing manifest.json)
    if path.is_dir():
        manifest_file = path / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Feature manifest not found in directory: {path}")
        store = MmapFeatureStore(manifest_file)
        print(f"Loaded {len(store)} image features via memory-mapped store.")
        return store

    if path.suffix.lower() == ".json":
        store = MmapFeatureStore(path)
        print(f"Loaded {len(store)} image features via memory-mapped store.")
        return store

    # Sibling manifest check (e.g., if pointing directly to features.npy inside a store directory)
    if path.suffix.lower() == ".npy" and (path.parent / "manifest.json").exists():
        store = MmapFeatureStore(path.parent / "manifest.json")
        print(f"Loaded {len(store)} image features via memory-mapped store.")
        return store

    # Legacy loader for dictionary-based .npy and .npz
    data = np.load(
        str(path),
        allow_pickle=True
    )
    if hasattr(data, "files"):
        features = {image_name: data[image_name] for image_name in data.files}
    elif hasattr(data, "item"):
        features = data.item()
    else:
        features = dict(data)

    print(f"Loaded {len(features)} image features.")
    return features


def extract_all_features(image_dir=None, model=None, save_path=None):
    if image_dir is None:
        image_dir = IMAGE_DIR
    if model is None:
        model = get_cnn_model()
    if save_path is None:
        save_path = FEATURE_FILE

    features = {}
    image_names = os.listdir(image_dir)
    print(f"Total Images: {len(image_names)}")

    for index, image_name in enumerate(image_names):
        image_path = os.path.join(image_dir, image_name)
        try:
            feature = extract_feature(image_path, model)
            feature = feature.reshape(-1, 2048)
            features[image_name] = feature
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

        if (index + 1) % 500 == 0:
            print(f"Processed {index+1}/{len(image_names)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, features)
    print(f"Features saved successfully to: {save_path}")
    return features


if __name__ == "__main__":
    print("Starting V2 CNN feature extraction...\n")
    extract_all_features()
