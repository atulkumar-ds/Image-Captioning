import os
import numpy as np

from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

IMAGE_DIR = "data/images"
FEATURE_FILE = "data/features/image_features.npz"



cnn_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

cnn_model.trainable = False



def extract_feature(image_path, model):

    img = image.load_img(
        image_path,
        target_size=(299, 299)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    features = model.predict(
        img_array,
        verbose=0
    )

    return features[0]



def extract_all_features(image_dir, model):

    features_dict = {}

    image_names = os.listdir(image_dir)

    image_names = [
        name for name in image_names
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total_images = len(image_names)

    for i, image_name in enumerate(image_names):

        image_path = os.path.join(
            image_dir,
            image_name
        )

        try:
            features = extract_feature(
                image_path,
                model
            )

            features_dict[image_name] = features

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == total_images:
            print(f"Processed {i + 1}/{total_images}")

    return features_dict




def save_features(features_dict, file_path):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    np.savez_compressed(
        file_path,
        **features_dict
    )

    print("\nFeatures saved successfully.")
    print("Location:", file_path)



def load_features(file_path):

    data = np.load(file_path)

    features_dict = {
        image_name: data[image_name]
        for image_name in data.files
    }

    return features_dict




if __name__ == "__main__":

    print("Starting CNN feature extraction...\n")

    features_dict = extract_all_features(
        IMAGE_DIR,
        cnn_model,
    )

    print("\nTotal Features Extracted:")
    print(len(features_dict))

    first_image = list(features_dict.keys())[0]

    print("\nExample Image:")
    print(first_image)

    print("\nFeature Shape:")
    print(features_dict[first_image].shape)

    print("\nFirst 10 Features:")
    print(features_dict[first_image][:10])

    save_features(
        features_dict,
        FEATURE_FILE
    )

    loaded_features = load_features(
        FEATURE_FILE
    )

    print("\nLoaded Features:")
    print(len(loaded_features))

    print("\nDone.")