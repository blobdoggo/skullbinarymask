import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms # Use v2 for better composability
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm # Progress bar

# --- Configuration ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 2 # Keep low if memory constrained, increase if possible
EPOCHS = 6
LEARNING_RATE = 1e-4 # Adam default is 1e-3, often 1e-4 is better for fine-tuning/segmentation
MODEL_SAVE_PATH = 'unet_pytorch_model.pth'

# --- Data Paths ---
BASE_DATA_DIR = "C:/Users/rushi/dataset" # Adjust if needed
TRAIN_IMG_DIR = os.path.join(BASE_DATA_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(BASE_DATA_DIR, "train/masks")
VAL_IMG_DIR = os.path.join(BASE_DATA_DIR, "val/images")
VAL_MASK_DIR = os.path.join(BASE_DATA_DIR, "val/masks")
PREDICT_IMG_PATH = os.path.join(BASE_DATA_DIR, "val/images/DSC00117.jpg") # Example image for prediction

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
        image_filenames = os.listdir("C:/Users/rushi/dataset/train/images")
        # extract on .jpg
        self.image_filenames = [f for f in image_filenames if f.endswith('.jpg')]
        mask_filenames = os.listdir("C:/Users/rushi/dataset/train/masks")
        # extract on .jpg
        self.mask_filenames = [f for f in mask_filenames if f.endswith('.jpg')]

        print(f"Found {len(self.image_filenames)} images and {len(self.mask_filenames)} masks in {image_dir} and {mask_dir} respectively.")

        # Basic check: Ensure corresponding images and masks exist
        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError("Number of images and masks do not match!")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Use PIL for image loading
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        return image, mask

# --- Transforms ---
# Define transforms for images and masks
# Normalization values (mean/std) for ImageNet are common, but calculate if needed for your specific dataset
# For masks, we only need resizing and converting to tensor (no normalization usually)
img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToImage(), # Converts PIL to tensor C x H x W
    transforms.ToDtype(torch.float32, scale=True), # Equivalent to ToTensor(), scales to [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Optional: Use ImageNet stats or calculate your own
])

mask_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST), # Use NEAREST for masks!
    transforms.ToImage(), # Converts PIL to tensor 1 x H x W
    transforms.ToDtype(torch.float32, scale=True), # Scales to [0.0, 1.0] - necessary for BCELoss
])

# --- DataLoaders ---
train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, img_transform, mask_transform)
val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, img_transform, mask_transform)

# Use num_workers > 0 to speed up data loading (uses background processes)
# Pin memory can sometimes speed up CPU->GPU transfer
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- U-Net Model Definition ---
class DoubleConv(nn.Module):
    """(Convolution => ReLU => Convolution => ReLU)"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # Use bias=False if using BatchNorm
            # nn.BatchNorm2d(mid_channels), # Optional: Add BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels), # Optional: Add BatchNorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Keras uses Conv2DTranspose with kernel=3, stride=2, padding='same'
            # To replicate this behavior more closely in PyTorch for doubling size:
            # kernel=3, stride=2, padding=1, output_padding=1
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv = DoubleConv(in_channels, out_channels) # Input channels = channels from up + skip connection


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2 size if necessary (due to potential rounding in pooling/transpose conv)
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # Concatenate along channel dimension
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False): # Set bilinear=False to use ConvTranspose
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        # Bottleneck equivalent (removed maxpool, added one more DoubleConv)
        # The original Keras code had a third conv block C3 before upsampling starts
        self.down3 = Down(256, 512) # Keras code had 256 -> 256 in bottleneck, this follows more standard UNet 256->512
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) # Bottleneck equivalent

        # Adjusting Up layers based on the new down path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear) # Add the last upsampling corresponding to inc
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # Added downsampling
        x5 = self.down4(x4) # Bottleneck features

        x = self.up1(x5, x4) # Upsample from bottleneck, concat with x4
        x = self.up2(x, x3)  # Upsample, concat with x3
        x = self.up3(x, x2)  # Upsample, concat with x2
        x = self.up4(x, x1)  # Upsample, concat with x1 (initial features)
        logits = self.outc(x)

        # Apply sigmoid for binary segmentation
        if self.n_classes == 1:
             return torch.sigmoid(logits)
        else:
             # Use Softmax for multi-class (or keep logits if using CrossEntropyLoss)
             return logits # Or nn.functional.softmax(logits, dim=1)

# --- Re-implementing the Keras U-Net structure more closely ---
# The previous UNet class is a more standard PyTorch UNet.
# Let's try to match the Keras layer structure exactly.
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

        # Bottleneck (matches Keras c3 block)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True) # c3

        # Decoder
        # Keras Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same")
        # PyTorch equivalent for doubling size with kernel=3, stride=2: padding=1, output_padding=1
        self.up_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # Adjusted output channels to match concat need (128)
        # Concatenation happens in forward pass
        self.conv_up1_1 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1) # Input channels = up_conv1_out (128) + conv2_2_out (128)
        self.relu_up1_1 = nn.ReLU(inplace=True)
        # Keras code had 256 filters here, mistake in my translation or Keras code? Let's match Keras provided code structure.
        # Re-evaluating Keras Decoder u1:
        # u1 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(c3) # Output: 256 channels
        # u1 = layers.concatenate([u1, c2]) # c2 is 128 channels. Concatenated = 256+128 = 384 channels
        # u1 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u1) # Output: 256 channels
        # Let's fix the PyTorch decoder based on this:
        self.up_conv1_fix = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1) # u1 - transpose conv
        # Concatenate c2 (128 channels) and up_conv1_fix (256 channels) -> 384 channels
        self.conv_up1_1_fix = nn.Conv2d(256 + 128, 256, kernel_size=3, padding=1) # u1 - conv after concat
        self.relu_up1_1_fix = nn.ReLU(inplace=True)

        # Keras Decoder u2:
        # u2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(u1) # Input u1 is 256 channels. Output: 128 channels
        # u2 = layers.concatenate([u2, c1]) # c1 is 64 channels. Concatenated = 128 + 64 = 192 channels
        # u2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2) # Output: 128 channels
        self.up_conv2_fix = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # u2 - transpose conv
        # Concatenate c1 (64 channels) and up_conv2_fix (128 channels) -> 192 channels
        self.conv_up2_1_fix = nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1) # u2 - conv after concat
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
        # p3 (bottleneck pool) is not used in decoder in the Keras code

        # Decoder (Fixed version)
        up1 = self.up_conv1_fix(c3) # Transpose conv: 256 channels
         # Ensure spatial dimensions match for concatenation
        diffY = c2.size()[2] - up1.size()[2]
        diffX = c2.size()[3] - up1.size()[3]
        up1 = nn.functional.pad(up1, [diffX // 2, diffX - diffX // 2,
                                       diffY // 2, diffY - diffY // 2])
        cat1 = torch.cat([up1, c2], dim=1) # Concat: 256 + 128 = 384 channels
        u1 = self.relu_up1_1_fix(self.conv_up1_1_fix(cat1)) # Conv: 256 channels

        up2 = self.up_conv2_fix(u1) # Transpose conv: 128 channels
         # Ensure spatial dimensions match for concatenation
        diffY = c1.size()[2] - up2.size()[2]
        diffX = c1.size()[3] - up2.size()[3]
        up2 = nn.functional.pad(up2, [diffX // 2, diffX - diffX // 2,
                                       diffY // 2, diffY - diffY // 2])
        cat2 = torch.cat([up2, c1], dim=1) # Concat: 128 + 64 = 192 channels
        u2 = self.relu_up2_1_fix(self.conv_up2_1_fix(cat2)) # Conv: 128 channels


        output = self.output_conv(u2)
        return self.output_activation(output)


# --- Initialize Model, Loss, Optimizer ---
# Choose which UNet version to use:
# model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
model = UNetKerasStyle(input_channels=3, output_channels=1).to(device)

criterion = nn.BCELoss() # Binary Cross Entropy for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2) # Optional: Learning rate scheduler

# --- Training Loop ---
history = {'train_loss': [], 'val_loss': []}

print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    running_train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")

    for i, (inputs, masks) in enumerate(train_loop):
        inputs, masks = inputs.to(device), masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        # Update progress bar
        train_loop.set_postfix(loss=loss.item()) # Show loss for the current batch


    epoch_train_loss = running_train_loss / len(train_loader)
    history['train_loss'].append(epoch_train_loss)

    # --- Validation Loop ---
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch")

    with torch.no_grad(): # Disable gradient calculations for validation
        for inputs, masks in val_loop:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            running_val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())

    epoch_val_loss = running_val_loss / len(val_loader)
    history['val_loss'].append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH + f".epoch{epoch+1}.pth") # Save model after each epoch
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Optional: Learning rate scheduling
    # scheduler.step(epoch_val_loss)

# --- Save the Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# --- Plot Loss Curves ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Training Loss')
plt.plot(range(1, EPOCHS + 1), history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (BCE)')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_pytorch.png')
print("Loss curve saved to loss_curve_pytorch.png")
# plt.show() # Uncomment to display the plot directly


# --- Prediction/Inference Example ---
print("\nRunning prediction example...")

# Load the trained model (if needed in a separate script/session)
# First, instantiate the model structure, then load the weights
# model_pred = UNet(n_channels=3, n_classes=1, bilinear=False).to(device) # Use the same structure as trained
model_pred = UNetKerasStyle(input_channels=3, output_channels=1).to(device)
model_pred.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model_pred.eval() # Set to evaluation mode

def preprocess_single_image(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0) # Add batch dimension (BCHW)
    return img

# Load and preprocess the prediction image
input_image_tensor = preprocess_single_image(PREDICT_IMG_PATH, img_transform).to(device)

# Perform prediction
with torch.no_grad():
    predicted_mask_tensor = model_pred(input_image_tensor)

# Post-process the prediction
predicted_mask = predicted_mask_tensor.squeeze().cpu().numpy() # Remove batch/channel dims, move to CPU, convert to numpy
predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8) # Threshold and convert to binary mask (0 or 1)

# Load original image for display
original_image = Image.open(PREDICT_IMG_PATH)

# Display original and predicted mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predicted Mask (PyTorch)")
plt.imshow(predicted_mask_binary, cmap="gray")
plt.axis('off')

plt.tight_layout()
plt.savefig('prediction_example_pytorch.png')
print(f"Prediction example saved to prediction_example_pytorch.png")
plt.show()

print("\nScript finished.")