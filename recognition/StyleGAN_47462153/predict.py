import os
import torch
from modules import Generator
from dataset import get_dataloader
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    predict()