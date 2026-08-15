import os
import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit(False)

from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

IMAGE_DIR = "../V1_Baseline_CNN_LSTM/data/images"

FEATURE_FILE = "../V2_Attention/model/v2_features.npy"
cnn_model = InceptionV3(
    weights="imagenet",
    include_top=False
)

cnn_model.trainable = False

print("CNN Output Shape:")
print(cnn_model.output_shape)

def extract_feature(image_path, model):

    img = image.load_img(
        image_path,
        target_size=(299, 299)
    )

    img = image.img_to_array(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    img = preprocess_input(img)

    features = model.predict(
        img,
        verbose=0
    )

    return features

def load_features(feature_file):

    features = np.load(
        feature_file,
        allow_pickle=True
    ).item()

    print(
        f"Loaded {len(features)} image features."
    )

    return features

def extract_all_features():

    features = {}

    image_names = os.listdir(
        IMAGE_DIR
    )

    print(
        f"Total Images: {len(image_names)}"
    )

    for index, image_name in enumerate(image_names):

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        feature = extract_feature(
            image_path,
            cnn_model
        )

        feature = feature.reshape(
            -1,
            2048
        )

        features[image_name] = feature

        if (index + 1) % 500 == 0:

            print(
                f"Processed {index+1}/{len(image_names)}"
            )

    np.save(
        FEATURE_FILE,
        features
    )

    print("\nFeatures Saved")
    print(FEATURE_FILE)


if __name__ == "__main__":

    extract_all_features()