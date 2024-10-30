import os
import torch
from modules import Generator
from dataset import get_dataloader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import traceback

# Set MPLCONFIGDIR to a writable directory to fix Matplotlib warning
os.environ['MPLCONFIGDIR'] = os.path.expanduser('~/.matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE_GEN = 0.001
LEARNING_RATE_DISC = 0.0005
BATCH_SIZE = 16
IMG_CHANNELS = 1
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
FIXED_IMAGE_SIZE = 64
DATA_ROOT = '/home/groups/comp3710/ADNI/AD_NC/train'
SAVE_MODEL = True
SAVE_MODEL_PATH = "./model_checkpoints"
SAVE_IMAGES_PATH = "./generated_images"
CHECKPOINT_FILE = "stylegan_checkpoint.pth.tar"
LOAD_MODEL = True
MAX_RUNTIME = 18 * 60
MAX_BATCHES_PER_EPOCH = 100
R1_GAMMA = 10

def compute_r1_loss(real_pred, real_img):
    """
    Computes R1 regularization for stability during discriminator training.
    """
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )[0]
    grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()
    return grad_penalty

def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to resume training in subsequent sessions.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, disc, gen_optimizer, disc_optimizer):
    """
    Loads a saved checkpoint, helping to resume training seamlessly.
    """
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
    disc.load_state_dict(checkpoint['disc_state_dict'], strict=False)
    try:
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
    except ValueError as e:
        print(f"Warning: {e}")
        print("Optimizers reinitialized due to parameter mismatch.")
    start_epoch = checkpoint['epoch']
    return start_epoch

def save_generated_images(generator, epoch, num_images=5):
    """
    Generates and saves sample images to visualize model progress.
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, Z_DIM).to(DEVICE)
        fake_images = generator(noise)
        fake_images = (fake_images * 0.5 + 0.5).cpu()
        for idx in range(num_images):
            img = fake_images[idx].squeeze()
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(SAVE_IMAGES_PATH, f"generated_epoch_{epoch}_img_{idx}.png"))
    generator.train()

def train_model(root_dir, batch_size, num_epochs, output_dir):
    """
    Trains the StyleGAN2 model using the provided arguments and parameters.
    """
    try:
        import train  # Import the train function from train.py
    except ImportError as e:
        print(f"Error importing train function: {e}")
        sys.exit(1)
    
    # Argument namespace for passing to the train function
    class Args:
        def __init__(self, data_root, batch_size, num_epochs, output_dir):
            self.data_root = data_root
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.output_dir = output_dir

    args = Args(root_dir, batch_size, num_epochs, output_dir)
    
    train.train(args)  # Call the train function

def generate_images(generator, seeds, outdir, truncation, device):
    """
    Generates synthetic images based on input seeds and saves them to disk.
    """
    os.makedirs(outdir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed)
            noise = torch.randn(1, Z_DIM).to(device)
            fake_image = generator(noise)
            fake_image = (fake_image.clamp(-1, 1) + 1) / 2
            save_path = os.path.join(outdir, f'generated_seed_{seed}.png')
            save_image(fake_image, save_path)
            print(f"Generated image saved at '{save_path}'")
    generator.train()

def load_test_images(test_dir, num_images, transform=None):
    """
    Loads test images from the specified directory for evaluation.
    """
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
    """
    Displays a grid of images to visually assess model outputs.
    """
    plt.figure(figsize=(15, 15))
    grid_size = int(np.ceil(np.sqrt(len(images))))
    for i, img in enumerate(images):
        plt.subplot(grid_size, grid_size, i+1)
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5)
        plt.imshow(img, cmap='gray' if IMG_CHANNELS == 1 else None)
        plt.axis('off')
    plt.suptitle(title, fontsize=20)
    plt.show()

def predict():
    """
    Main function to load a trained model, generate images, and visualize results.
    """
    parser = argparse.ArgumentParser(description='Generate and Visualize Images using Trained StyleGAN2 Generator')
    parser.add_argument('--checkpoint_dir', type=str, default=SAVE_MODEL_PATH, help='Directory where models are saved')
    parser.add_argument('--seeds', type=str, default='0-9', help='Comma-separated list of seeds or ranges (e.g., "0-4,10,15-20")')
    parser.add_argument('--outdir', type=str, default=SAVE_IMAGES_PATH, help='Directory to save generated images')
    parser.add_argument('--truncation', type=float, default=0.7, help='Truncation psi value')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--test_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC/test', help='Path to test data directory')
    parser.add_argument('--num_test_images', type=int, default=5, help='Number of test images to load and visualize')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='Path to training data (for training if model not found)')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs (for training if model not found)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size (for training if model not found)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    generator = Generator(z_dim=Z_DIM, w_dim=W_DIM, in_channels=IN_CHANNELS, img_channels=IMG_CHANNELS).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, CHECKPOINT_FILE)
    gen_final_path = os.path.join(args.checkpoint_dir, "generator_final.pth")

    if os.path.exists(gen_final_path):
        try:
            generator.load_state_dict(torch.load(gen_final_path, map_location=device))
            print(f"Loaded generator model from '{gen_final_path}'")
        except RuntimeError as e:
            print(f"Error loading generator_final.pth: {e}")
            sys.exit(1)
    else:
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                generator.load_state_dict(checkpoint['gen_state_dict'])
                print(f"Loaded generator model from checkpoint '{checkpoint_path}'")
            except (KeyError, RuntimeError) as e:
                print(f"Error loading generator model from checkpoint: {e}")
                sys.exit(1)
        else:
            print("Generator model not found. Training the model...")
            train_model(args.data_root, args.batch_size, args.num_epochs, args.checkpoint_dir)
            if os.path.exists(gen_final_path):
                try:
                    generator.load_state_dict(torch.load(gen_final_path, map_location=device))
                    print(f"Loaded generator model from '{gen_final_path}' after training")
                except RuntimeError as e:
                    print(f"Error loading generator_final.pth after training: {e}")
                    sys.exit(1)
            else:
                print("Failed to save generator_final.pth after training.")
                sys.exit(1)

    seed_input = args.seeds.split(',')
    seeds = []
    for part in seed_input:
        if '-' in part:
            start, end = part.split('-')
            seeds.extend(range(int(start), int(end)+1))
        else:
            seeds.append(int(part))

    generate_images(generator, seeds, args.outdir, args.truncation, device)

    test_transform = transforms.Compose([
        transforms.Resize(FIXED_IMAGE_SIZE),
        transforms.CenterCrop(FIXED_IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=IMG_CHANNELS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    test_images = load_test_images(args.test_dir, args.num_test_images, transform=test_transform)

    visualize_images(test_images, title="Test Images")
    print("Prediction and visualization completed.")

if __name__ == "__main__":
    try:
        predict()
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        sys.exit(1)
