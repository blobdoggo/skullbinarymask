
import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)
from keras import models
print("models")
import numpy as np
print("numpy")
model = models.load_model("my_unet_model.keras")  # Load the trained model
print("model loaded")
test_image_path = "C:/Users/rushi/dataset/val/images/DSC00117.jpg"

def preprocess_test(test_image_path, target_size=(256, 256)):
    img = keras.utils.load_img(test_image_path, target_size=target_size)
    img = keras.utils.img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

input_image = preprocess_test(test_image_path)
# Generate mask
predicted_mask = model.predict(input_image)[0]  # Remove batch dimension
# Convert mask to binary (Thresholding)
predicted_mask = tf.image.resize(predicted_mask, (5304,7592))  # Resize to original image size
predicted_mask = (predicted_mask > 0.5).numpy().astype(np.uint8)  # Convert to binary mask


import matplotlib.pyplot as plt

# Load original image for comparison
original_image = keras.utils.load_img(test_image_path)

# Display original and predicted mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask.squeeze(), cmap="gray")  # Remove extra dimensions
plt.show()