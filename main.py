import argparse
import os
import sys
from saveallmasks import process_images
import torch
from unettorchnosplit import UNetKerasStyle

input_folder = 'dataset/input'
output_folder = 'dataset/output'

def setup_environment():
    # Placeholder for any environment setup (e.g., logging, seeds)
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Skull Binary Mask Generator")
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input folder (default: ./dataset/input)')
    parser.add_argument('--output', '-o', type=str,
                        help='Path to the output folder (default: ./dataset/output)')
    return parser.parse_args()

def ensure_folder(path, create_if_not_exists=True):
    if not os.path.exists(path) and create_if_not_exists:
        os.makedirs(path)
    else:
        if not os.path.exists(path):
            print(f"Error: The specified path does not exist: {path}")
            sys.exit(1)

def main():
    global input_folder
    global output_folder

    setup_environment()
    args = parse_args()

    base_folder = os.path.abspath('dataset')
    input_folder = os.path.abspath(args.input) if args.input else os.path.join(base_folder, 'input')
    output_folder = os.path.abspath(args.output) if args.output else os.path.join(base_folder, 'output')

    ensure_folder(input_folder, False)
    ensure_folder(output_folder)

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Main program logic goes here

if __name__ == "__main__":
    main()

    # ==== Config ====
    MODEL_SAVE_PATH = 'skull_model.pth'
    #INPUT_FOLDER = 'H:/dataset/Test1'
    #OUTPUT_FOLDER = 'H:/dataset/Test1Masks'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Create output directory if it doesn't exist ====
    os.makedirs(output_folder, exist_ok=True)

    # ==== Load Model ====
    model_pred = UNetKerasStyle(input_channels=3, output_channels=1).to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        model_pred.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model_pred.eval()
    else:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Cannot run prediction.")
        exit()




    # ==== Loop through all images ====
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in {input_folder}")
        exit()



    # Call the subroutine
    process_images(
        image_files=image_files,
        input_folder=input_folder,
        output_folder=output_folder,
        model=model_pred,
        device=DEVICE
    )