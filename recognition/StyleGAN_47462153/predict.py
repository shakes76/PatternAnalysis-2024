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
    pass

if __name__ == "__main__":
    predict()