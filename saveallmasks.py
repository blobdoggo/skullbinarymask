print("\nRunning batch prediction on folder...")
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
from unettorchnosplit import UNetKerasStyle, img_transform

# ==== Config ====
MODEL_SAVE_PATH = 'unet_pytorch_split_model.pth'
INPUT_FOLDER = 'H:/dataset/Test1'
OUTPUT_FOLDER = 'C:/Users/rushi/dataset/Test1Masks'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Create output directory if it doesn't exist ====
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== Load Model ====
model_pred = UNetKerasStyle(input_channels=3, output_channels=1).to(DEVICE)
if os.path.exists(MODEL_SAVE_PATH):
    model_pred.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model_pred.eval()
else:
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Cannot run prediction.")
    exit()

# ==== Preprocess ====
def preprocess(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    return img

# ==== Loop through all images ====
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"No images found in {INPUT_FOLDER}")
    exit()

for filename in image_files:
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_mask_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_mask.jpg")

    try:
        input_tensor = preprocess(input_path, img_transform).to(DEVICE)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

    with torch.no_grad():
        predicted_mask_tensor = model_pred(input_tensor)

    predicted_mask = predicted_mask_tensor.squeeze().cpu().numpy()
    
    # Load original image size to match mask to it
    original_image = Image.open(input_path)
    original_size = original_image.size  # (width, height)

    # Resize prediction to original size
    predicted_mask_resized = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_LINEAR)
    predicted_mask_binary = (predicted_mask_resized > 0.5).astype(np.uint8) * 255

    # Save binary mask
    cv2.imwrite(output_mask_path, predicted_mask_binary)
    print(f"Saved mask for {filename} to {output_mask_path}")

print("\nBatch prediction complete.")
