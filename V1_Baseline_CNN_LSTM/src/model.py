from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Add
)


# Build Image Captioning Model

def build_caption_model(
    vocab_size,
    max_length,
    feature_size=2048,
    embedding_dim=256,
    hidden_dim=256
):

    # 1. IMAGE BRANCH

    # Input from InceptionV3
    # Shape: (2048,)
    image_input = Input(
        shape=(feature_size,),
        name="image_input"
    )

    # Regularization
    image_features = Dropout(
        0.5,
        name="image_dropout"
    )(image_input)

    # Convert image representation:
    # 2048 -> 256
    image_features = Dense(
        hidden_dim,
        activation="relu",
        name="image_dense"
    )(image_features)


    # 2. CAPTION BRANCH

    # Input:
    # padded caption sequence
    # Shape: (max_length,)
    caption_input = Input(
        shape=(max_length,),
        name="caption_input"
    )

    # Word indices -> word vectors
    caption_features = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="word_embedding"
    )(caption_input)

    caption_features = Dropout(
        0.5,
        name="caption_dropout"
    )(caption_features)

    # Process caption sequence
    caption_features = LSTM(
        hidden_dim,
        name="caption_lstm"
    )(caption_features)


    # 3. IMAGE + CAPTION FUSION

    merged = Add(
        name="image_caption_fusion"
    )([
        image_features,
        caption_features
    ])


    # 4. JOINT REPRESENTATION

    merged = Dense(
        hidden_dim,
        activation="relu",
        name="fusion_dense"
    )(merged)


    # 5. NEXT WORD PREDICTION

    output = Dense(
        vocab_size,
        activation="softmax",
        name="word_output"
    )(merged)


    # 6. CREATE MODEL

    model = Model(
        inputs=[
            image_input,
            caption_input
        ],
        outputs=output,
        name="ImageCaptioningModel"
    )

    return model


# Compile Model

def compile_caption_model(model):

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Test Model

if __name__ == "__main__":

    # Temporary values ONLY for testing model.py
    vocab_size = 3000
    max_length = 34

    print("\nBuilding Image Captioning Model...\n")

    model = build_caption_model(
        vocab_size=vocab_size,
        max_length=max_length,
        feature_size=2048,
        embedding_dim=256,
        hidden_dim=256
    )

    model = compile_caption_model(model)

    model.summary()

    print("\nModel built successfully.")