print("\nRunning batch prediction on folder...")
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
from unettorchnosplit import UNetKerasStyle, img_transform

# ==== Preprocess ====
def preprocess(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def process_single_image(filename, input_folder, output_folder, model, device):
    image_path = os.path.join(input_folder, filename)
    img = preprocess(image_path, img_transform).to(device)

    with torch.no_grad():
        output = model(img)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()  # Binarize the output

    # Convert to numpy and save
    output_np = output.squeeze().cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)  # Scale to [0, 255]
    
    # Save the mask
    mask_filename = os.path.splitext(filename)[0] + '_mask.png'
    mask_path = os.path.join(output_folder, mask_filename)
    cv2.imwrite(mask_path, output_np)

    print(f"Processed {filename} -> {mask_filename}")

def process_image(filename, input_folder, output_folder, model_pred, DEVICE):
    
    input_path = os.path.join(input_folder, filename)
    output_mask_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask.jpg")

    try:
        input_tensor = preprocess(input_path, img_transform).to(DEVICE)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        return

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


def process_images(image_files, input_folder, output_folder, model, device):
    for filename in image_files:
        process_image(filename, input_folder, output_folder, model, device)
    print("\nBatch prediction complete.")

