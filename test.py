import os
import numpy as np

# --- Data Paths ---
BASE_DATA_DIR = "C:/Users/rushi/dataset" # Adjust if needed
TRAIN_IMG_DIR = os.path.join(BASE_DATA_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(BASE_DATA_DIR, "train/masks")
VAL_IMG_DIR = os.path.join(BASE_DATA_DIR, "val/images")
VAL_MASK_DIR = os.path.join(BASE_DATA_DIR, "val/masks")
PREDICT_IMG_PATH = os.path.join(BASE_DATA_DIR, "val/images/DSC00117.jpg") # Example image for prediction



print(f"Found {len(image_filenames)} images and {len(mask_filenames)} masks.")