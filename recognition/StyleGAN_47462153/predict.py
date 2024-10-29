import os
import torch
from modules import Generator
from dataset import get_dataloader
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def generate_images(generator, seeds, outdir, truncation, device):
    os.makedirs(outdir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed)
            noise = torch.randn(1, 512).to(device)
            fake_image = generator(noise)
            fake_image = (fake_image.clamp(-1, 1) + 1) / 2
            save_path = os.path.join(outdir, f'generated_seed_{seed}.png')
            save_image(fake_image, save_path)
            print(f"Generated image saved at '{save_path}'")

def predict():
    parser = argparse.ArgumentParser(description='Generate and Visualize Images using Trained StyleGAN2 Generator')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained generator checkpoint')
    parser.add_argument('--seeds', type=str, default='0-9', help='Comma-separated list of seeds or ranges (e.g., "0-4,10,15-20")')
    parser.add_argument('--outdir', type=str, default='generated_images', help='Directory to save generated images')
    parser.add_argument('--truncation', type=float, default=0.7, help='Truncation psi value')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu"')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    generator = Generator(z_dim=512, w_dim=512, in_channels=512, img_channels=1).to(device)

    if os.path.exists(args.checkpoint):
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['gen_state_dict'])
            print(f"Loaded generator checkpoint from '{args.checkpoint}'")
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            print("Exiting...")
            exit(1)
    else:
        raise FileNotFoundError(f"Checkpoint file '{args.checkpoint}' not found.")

def load_test_images(test_dir, num_images, transform=None):
    pass

def visualize_images(images, title="Generated Images"):
    pass

if __name__ == "__main__":
    predict()