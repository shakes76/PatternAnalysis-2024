import os
import torch
from modules import Generator
from dataset import get_dataloader
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms

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

def load_test_images(test_dir, num_images, transform=None):
    test_image_paths = []
    for subdir in ['AD', 'NC']:
        subdir_path = os.path.join(test_dir, subdir)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    test_image_paths.append(os.path.join(subdir_path, file_name))
                    if len(test_image_paths) >= num_images:
                        break
        if len(test_image_paths) >= num_images:
            break

    images = []
    for img_path in test_image_paths:
        img = Image.open(img_path).convert("RGB")
        if transform:
            img = transform(img)
        images.append(img)
    return images

def visualize_images(images, title="Generated Images"):
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(images):
        plt.subplot(5, 5, i+1)
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title, fontsize=20)
    plt.show()

def predict():
    parser = argparse.ArgumentParser(description='Generate and Visualize Images using Trained StyleGAN2 Generator')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained generator checkpoint')
    parser.add_argument('--seeds', type=str, default='0-9', help='Comma-separated list of seeds or ranges (e.g., "0-4,10,15-20")')
    parser.add_argument('--outdir', type=str, default='generated_images', help='Directory to save generated images')
    parser.add_argument('--truncation', type=float, default=0.7, help='Truncation psi value')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--test_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC/test', help='Path to test data directory')
    parser.add_argument('--num_test_images', type=int, default=5, help='Number of test images to load and visualize')
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

    seed_input = args.seeds.split(',')
    seeds = []
    for part in seed_input:
        if '-' in part:
            start, end = part.split('-')
            seeds.extend(range(int(start), int(end)+1))
        else:
            seeds.append(int(part))

    generate_images(
        generator=generator,
        seeds=seeds,
        outdir=args.outdir,
        truncation=args.truncation,
        device=device
    )

    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    test_images = load_test_images(args.test_dir, args.num_test_images, transform=test_transform)

    visualize_images(test_images, title="Test Images")

if __name__ == "__main__":
    predict()