# model.py

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.models import Model

# Input shape
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 224, 224, 3

def build_dual_channel_model():
    # Shared input layer
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    # First channel: InceptionV3
    inception_base = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_layer)
    inception_out = GlobalAveragePooling2D()(inception_base.output)

    # Second channel: Xception
    xception_base = Xception(weights='imagenet', include_top=False, input_tensor=input_layer)
    xception_out = GlobalAveragePooling2D()(xception_base.output)

    # Concatenate the outputs of both channels
    merged = Concatenate()([inception_out, xception_out])

    # Add a fully connected layer and output layer
    dense = Dense(256, activation='relu')(merged)
    output = Dense(4, activation='softmax')(dense)  # 4 classes: glioma, meningioma, notumor, pituitary

    # Define the model with a single input
    model = Model(inputs=input_layer, outputs=output)

    return model

if __name__ == "__main__":
    model = build_dual_channel_model()
    model.summary()
