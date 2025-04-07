# -*- coding: utf-8 -*-


import tensorflow as tf
import keras as keras
from keras import layers, models
import numpy as np
import os

# Image size
IMG_SIZE = (256, 256)
# Load images and masks
def load_data(image_dir, mask_dir):
    images, masks = [], []
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    for img_name, mask_name in zip(image_filenames, mask_filenames):
        img = keras.utils.load_img(os.path.join(image_dir, img_name), target_size=IMG_SIZE)
        mask = keras.utils.load_img(os.path.join(mask_dir, mask_name), target_size=IMG_SIZE, color_mode="grayscale")

        img = keras.utils.img_to_array(img) / 255.0  # Normalize
        mask = keras.utils.img_to_array(mask) / 255.0  # Normalize

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load dataset
image_dir = "C:/Users/rushi/dataset/train/images"
mask_dir = "C:/Users/rushi/dataset/train/masks"
img_val = "C:/Users/rushi/dataset/val/images"
mask_val = "C:/Users/rushi/dataset/val/masks"

# Split into training and validation sets
X_train, Y_train = load_data(image_dir, mask_dir)
X_val, Y_val = load_data(img_val, mask_val)

def unet_model(input_size=(256, 256, 3)):

    inputs = layers.Input(input_size)

    # Encoder (Downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    #bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    
    # Decoder (Upsampling)
    u1 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(c3)
    u1 = layers.concatenate([u1, c2])
    u1 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u1)

    u2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(u1)
    u2 = layers.concatenate([u2, c1])
    u2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2)

   

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(u2)  # Single-class mask output

    model = models.Model(inputs, outputs)
    return model

# Compile model
model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Train model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=2, epochs=6)

# prompt: save the above model

model.save('my_unet_model.keras')

model = models.load_model("my_unet_model.keras")  # Load the trained model

def preprocess_image(image_path, target_size=(256, 256)):
    img = keras.utils.load_img(image_path, target_size=target_size)
    img = keras.utils.img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load and preprocess a new image
image_path = "C:/Users/rushi/dataset/val/images/DSC00117.jpg"
input_image = preprocess_image(image_path)
# Generate mask
predicted_mask = model.predict(input_image)[0]  # Remove batch dimension
# Convert mask to binary (Thresholding)
predicted_mask = (predicted_mask > 0.5)
predicted_mask = predicted_mask.astype(np.uint8)  # Convert to binary mask

import matplotlib.pyplot as plt

# Load original image for comparison
original_image = keras.utils.load_img(image_path)

# Display original and predicted mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask.squeeze(), cmap="gray")  # Remove extra dimensions
plt.show()