import os
import numpy as np

from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

IMAGE_DIR = "data/images"
FEATURE_FILE = "data/features/image_features.npy"

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

if __name__ == "__main__":

    image_path = "data/images/1000268201_693b08cb0e.jpg"

    features = extract_feature(
        image_path,
        cnn_model
    )

    print("\nFeature Shape:")
    print(features.shape)