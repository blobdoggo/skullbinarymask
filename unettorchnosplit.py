# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms.v2 as transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import multiprocessing # Import multiprocessing

# --- Configuration ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 6
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'unet_pytorch_split_model.pth'
VALIDATION_SPLIT = 0.2 # Use 20% of the data for validation
RANDOM_SEED = 42 # For reproducible splits

# --- Data Paths (Single Source) ---
DATA_DIR = "C:/Users/rushi/dataset" # Adjust if needed
ALL_IMG_DIR = os.path.join(DATA_DIR, "train/images") # CHANGE THIS to your single image folder
ALL_MASK_DIR = os.path.join(DATA_DIR, "train/masks") # CHANGE THIS to your single mask folder

# --- Ensure Prediction Image Path is Set Correctly ---
# Make sure this image exists within your ALL_IMG_DIR
# Using the uppercase version as it likely exists in the image folder
PREDICT_IMG_PATH = os.path.join(ALL_IMG_DIR, "DSC00117.jpg") # Example, change to a real image name

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch compiled with CUDA version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


# --- Dataset Definition ---
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform, mask_transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
             raise FileNotFoundError(f"Image directory '{image_dir}' or mask directory '{mask_dir}' not found.")

        # Store full paths directly or keep filenames and join later
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

        if not self.image_filenames:
            print(f"Warning: No image files found in {image_dir}")
        if not self.mask_filenames:
             print(f"Warning: No mask files found in {mask_dir}")

        if len(self.image_filenames) != len(self.mask_filenames):
            print(f"Warning: Number of images ({len(self.image_filenames)}) and masks ({len(self.mask_filenames)}) do not match.")
            # Decide how critical this is - maybe proceed or raise error depending on use case
            # raise ValueError("Number of images and masks must match for reliable pairing.")

        # Build a mapping from image basename (lowercase) to mask filename
        # This is more robust than relying on sorted lists when names don't match exactly
        self.mask_map = {}
        for mask_fname in self.mask_filenames:
            mask_basename = os.path.splitext(mask_fname)[0]
            # Store mask filename associated with its lowercase basename (without suffix if needed)
            # Handle potential "_mask" suffix variations case-insensitively
            processed_basename = mask_basename.lower()
            if processed_basename.endswith("_mask"):
                 processed_basename = processed_basename[:-5] # Remove "_mask"
            self.mask_map[processed_basename] = mask_fname # map 'dsc...' -> 'dsc..._mask.png'

        # Verification (optional but recommended)
        num_matched = 0
        unmatched_images = []
        for img_fname in self.image_filenames:
            img_basename_lower = os.path.splitext(img_fname)[0].lower()
            if img_basename_lower in self.mask_map:
                num_matched += 1
            else:
                unmatched_images.append(img_fname)

        print(f"Dataset Verification: Found {len(self.image_filenames)} images and {len(self.mask_filenames)} masks.")
        print(f"Successfully matched {num_matched} image-mask pairs based on case-insensitive basename.")
        if unmatched_images:
             print(f"Warning: Could not find corresponding masks for {len(unmatched_images)} images:")
             # Print first few unmatched for debugging
             print(unmatched_images[:10])
             # Depending on severity, you might want to raise an error here
             # raise ValueError("Could not match all images to masks.")


    def __len__(self):
        # Length should be based on images we can actually find a mask for
        # Or simply the number of images, and __getitem__ handles mismatches
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if idx >= len(self.image_filenames):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.image_filenames)}")

        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Find corresponding mask using the pre-built map
        img_basename_lower = os.path.splitext(img_name)[0].lower()

        if img_basename_lower in self.mask_map:
            mask_name = self.mask_map[img_basename_lower]
            mask_path = os.path.join(self.mask_dir, mask_name)
        else:
            # Handle case where no mask was found during init verification
            raise FileNotFoundError(f"Internal Error: Could not find corresponding mask for image {img_name} (basename: {img_basename_lower}) in pre-built map. Check init verification.")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError as e:
             print(f"Error opening file during getitem: {e}")
             raise e
        except Exception as e:
             print(f"Error processing image {img_path} or mask {mask_path}: {e}")
             raise e

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        return image, mask

# --- Transforms (No changes needed) ---
img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

# --- U-Net Model Definition (UNetKerasStyle) ---
# (Model definition code remains the same - place it here)
class UNetKerasStyle(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        # Encoder
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # p1
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # p2
        # Bottleneck
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True) # c3
        # Decoder
        self.up_conv1_fix = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_up1_1_fix = nn.Conv2d(256 + 128, 256, kernel_size=3, padding=1)
        self.relu_up1_1_fix = nn.ReLU(inplace=True)
        self.up_conv2_fix = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_up2_1_fix = nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1)
        self.relu_up2_1_fix = nn.ReLU(inplace=True)
        # Output layer
        self.output_conv = nn.Conv2d(128, output_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x))))
        p1 = self.pool1(c1)
        c2 = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(p1))))
        p2 = self.pool2(c2)
        c3 = self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(p2))))
        # Decoder
        up1 = self.up_conv1_fix(c3)
        diffY = c2.size()[2] - up1.size()[2]; diffX = c2.size()[3] - up1.size()[3]
        up1 = nn.functional.pad(up1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        cat1 = torch.cat([up1, c2], dim=1)
        u1 = self.relu_up1_1_fix(self.conv_up1_1_fix(cat1))
        up2 = self.up_conv2_fix(u1)
        diffY = c1.size()[2] - up2.size()[2]; diffX = c1.size()[3] - up2.size()[3]
        up2 = nn.functional.pad(up2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        cat2 = torch.cat([up2, c1], dim=1)
        u2 = self.relu_up2_1_fix(self.conv_up2_1_fix(cat2))
        output = self.output_conv(u2)
        return self.output_activation(output)


# ==============================================================================
# Main execution block - REQUIRED for multiprocessing on Windows
# ==============================================================================
if __name__ == '__main__':
    # Fix for multiprocessing spawn error on Windows
    multiprocessing.freeze_support()

    # --- Create Full Dataset ---
    try:
        full_dataset = SegmentationDataset(ALL_IMG_DIR, ALL_MASK_DIR, img_transform, mask_transform)
        print(f"Successfully initialized dataset handler. Total images found: {len(full_dataset)}")
        if len(full_dataset) == 0:
            print("Error: Dataset is empty or no matches found. Check image/mask paths and contents.")
            exit()
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error initializing dataset: {e}")
        print("Please check your ALL_IMG_DIR and ALL_MASK_DIR paths and ensure they contain corresponding image and mask files.")
        exit()

    # --- Split Dataset ---
    num_samples = len(full_dataset)
    if num_samples < 2 : # Need at least one sample for train and val
        print("Error: Not enough samples in the dataset to perform a train/validation split.")
        exit()

    num_val = int(np.floor(VALIDATION_SPLIT * num_samples))
    # Ensure num_val is at least 1 if VALIDATION_SPLIT > 0 and num_samples > 0
    if VALIDATION_SPLIT > 0 and num_val == 0 and num_samples > 0:
        num_val = 1
    num_train = num_samples - num_val

    if num_train == 0:
        print(f"Warning: Validation split ({VALIDATION_SPLIT}) resulted in 0 training samples. Adjust split or add data.")
        # Decide how to handle: maybe use all data for training? For now, exit.
        exit()

    print(f"Splitting dataset: {num_train} training samples, {num_val} validation samples")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(full_dataset, [num_train, num_val], generator=generator)

    # --- DataLoaders ---
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=default_collate,
                              persistent_workers=True if torch.cuda.is_available() else False) # Keep workers alive
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=default_collate,
                            persistent_workers=True if torch.cuda.is_available() else False)

    # --- Initialize Model, Loss, Optimizer ---
    model = UNetKerasStyle(input_channels=3, output_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training and Validation Loops ---
    history = {'train_loss': [], 'val_loss': []}

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")

        for i, (inputs, masks) in enumerate(train_loop):
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        if len(train_loader) > 0:
            epoch_train_loss = running_train_loss / len(train_loader)
        else:
            epoch_train_loss = 0.0
            print("Warning: train_loader is empty.")
        history['train_loss'].append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch")

        with torch.no_grad():
            for inputs, masks in val_loop:
                inputs = inputs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        if len(val_loader) > 0:
            epoch_val_loss = running_val_loss / len(val_loader)
        else:
            epoch_val_loss = 0.0
            print("Warning: val_loader is empty.")
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Save the model after each epoch (overwrites previous)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model checkpoint saved to {MODEL_SAVE_PATH}")


    print("Training finished.")

    # --- Plot Loss Curves ---
    print("Plotting loss curves...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_pytorch_split.png')
    print("Loss curve saved to loss_curve_pytorch_split.png")
    # plt.show()

    # --- Prediction/Inference Example ---
    print("\nRunning prediction example...")

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