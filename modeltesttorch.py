print("\nRunning prediction example...")
from unettorchnosplit import UNetKerasStyle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from unettorchnosplit import img_transform
import cv2
MODEL_SAVE_PATH = 'unet_pytorch_split_model.pth'
PREDICT_IMG_PATH = 'C:/Users/rushi/dataset/train/images/DSC00117.jpg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pred = UNetKerasStyle(input_channels=3, output_channels=1).to(device)
if os.path.exists(MODEL_SAVE_PATH):
    model_pred.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model_pred.eval()
else:
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Cannot run prediction.")
    exit()

def preprocess_single_image(image_path, transform):
    if not os.path.exists(image_path):
            raise FileNotFoundError(f"Prediction image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    return img

try:
    input_image_tensor = preprocess_single_image(PREDICT_IMG_PATH, img_transform).to(device)
except FileNotFoundError as e:
    print(e)
    print("Cannot perform prediction.")
    exit()

with torch.no_grad():
    predicted_mask_tensor = model_pred(input_image_tensor)

predicted_mask = predicted_mask_tensor.squeeze().cpu().numpy()  
predicted_mask = cv2.resize(predicted_mask, (7952, 5304), interpolation=cv2.INTER_LINEAR)  # Resize to original image size
predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

try:
    original_image = Image.open(PREDICT_IMG_PATH)
except FileNotFoundError:
    print(f"Could not load original image for display: {PREDICT_IMG_PATH}")
    original_image = None

plt.figure(figsize=(12, 6))
if original_image:
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
else:
        plt.subplot(1, 1, 1) # Only one plot if original is missing

plt.title("Predicted Mask (PyTorch)")
plt.imshow(predicted_mask_binary, cmap="gray")
plt.axis('off')

plt.tight_layout()
plt.savefig('prediction_example_pytorch_split.png')
print(f"Prediction example saved to prediction_example_pytorch_split.png")
plt.show()

print("\nScript finished.")