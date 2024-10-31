import time
import os
import torch
from modules import Generator, Discriminator
from dataset import get_dataloader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sys
import traceback
from sklearn.manifold import TSNE
from torchvision import models
import numpy as np

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

# Redirect cache directory to a user-writable location
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

def compute_r1_loss(real_pred, real_img):
    """
    Computes the R1 regularization loss for better stability in training.
    """
    grad_real = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()
    return grad_penalty

def save_checkpoint(state, filename):
    """
    Saves the model and optimizer states to resume training later.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, disc, gen_optimizer, disc_optimizer):
    """
    Loads a saved training checkpoint to continue from where it left off.
    """
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
    disc.load_state_dict(checkpoint['disc_state_dict'], strict=False)
    try:
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
    except ValueError as e:
        print(f"Warning: {e}")
        print("Optimizer states couldn't be loaded due to parameter mismatch.")
    return checkpoint['epoch']

def save_generated_images(generator, epoch, num_images=5):
    """
    Generates and saves a few images after each epoch for monitoring.
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

def plot_embeddings(dataloader, labels, output_path):
    """
    Computes and plots t-SNE embeddings of real images to visualize clustering.
    """
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet = resnet.to(DEVICE)
    resnet.eval()

    features = []
    image_labels = []

    with torch.no_grad():
        for real, label in dataloader:
            real = real.to(DEVICE)
            if IMG_CHANNELS == 1:
                real = real.repeat(1, 3, 1, 1)
            feat = resnet(real)
            features.append(feat.cpu().numpy())
            image_labels.extend(label.cpu().numpy())

    features = np.concatenate(features, axis=0)
    image_labels = np.array(image_labels)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=image_labels, cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Embeddings of Real Images")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(output_path)
    plt.close()

def train(args):
    try:
        gen = Generator(z_dim=Z_DIM, w_dim=W_DIM, in_channels=IN_CHANNELS, img_channels=IMG_CHANNELS).to(DEVICE)
        disc = Discriminator(in_channels=IN_CHANNELS, img_channels=IMG_CHANNELS).to(DEVICE)

        # Initialize optimizers with specific learning rates and betas
        gen_optimizer = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0, 0.99), eps=1e-8)
        disc_optimizer = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0, 0.99), eps=1e-8)

        scaler = GradScaler()

        checkpoint_path = os.path.join(args.output_dir, CHECKPOINT_FILE)
        if LOAD_MODEL and os.path.exists(checkpoint_path):
            print("Checkpoint found. Loading...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            start_epoch = load_checkpoint(checkpoint, gen, disc, gen_optimizer, disc_optimizer)
        else:
            start_epoch = 0
            print("Starting training from scratch")

        gen.train()
        disc.train()

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(SAVE_IMAGES_PATH, exist_ok=True)

        dataloader, dataset = get_dataloader(FIXED_IMAGE_SIZE, args.batch_size, args.data_root)
        num_epochs = args.num_epochs
        start_time = time.time()

        gen_losses, disc_losses = [], []

        for epoch in range(start_epoch, num_epochs):
            gen_loss_accum = 0.0
            disc_loss_accum = 0.0

            for batch_idx, (real, labels) in enumerate(dataloader):
                if batch_idx >= MAX_BATCHES_PER_EPOCH:
                    break

                real = real.to(DEVICE)
                batch_size_current = real.size(0)
                noise = torch.randn(batch_size_current, Z_DIM).to(DEVICE)

                # Training the discriminator with R1 regularization and mixed precision
                disc_optimizer.zero_grad()
                with autocast():
                    fake = gen(noise)
                    real.requires_grad_()
                    real_pred = disc(real)
                    fake_pred = disc(fake.detach())

                    d_loss_real = F.softplus(-real_pred).mean()
                    d_loss_fake = F.softplus(fake_pred).mean()
                    d_loss = d_loss_real + d_loss_fake
                    r1_loss = compute_r1_loss(real_pred, real)
                    disc_loss = d_loss + (R1_GAMMA / 2) * r1_loss

                scaler.scale(disc_loss).backward()
                scaler.step(disc_optimizer)
                scaler.update()
                disc_loss_accum += disc_loss.item()

                # Training the generator to maximize the discriminator's error
                gen_optimizer.zero_grad()
                with autocast():
                    fake = gen(noise)
                    fake_pred = disc(fake)
                    gen_loss = F.softplus(-fake_pred).mean()

                scaler.scale(gen_loss).backward()
                scaler.step(gen_optimizer)
                scaler.update()
                gen_loss_accum += gen_loss.item()

                elapsed_time = time.time() - start_time
                if elapsed_time > MAX_RUNTIME:
                    print("Max runtime reached, saving checkpoint.")
                    save_checkpoint({
                        'gen_state_dict': gen.state_dict(),
                        'disc_state_dict': disc.state_dict(),
                        'gen_optimizer': gen_optimizer.state_dict(),
                        'disc_optimizer': disc_optimizer.state_dict(),
                        'epoch': epoch,
                    }, os.path.join(args.output_dir, CHECKPOINT_FILE))
                    return

            avg_gen_loss = gen_loss_accum / len(dataloader)
            avg_disc_loss = disc_loss_accum / len(dataloader)
            gen_losses.append(avg_gen_loss)
            disc_losses.append(avg_disc_loss)

            if SAVE_MODEL:
                save_checkpoint({
                    'gen_state_dict': gen.state_dict(),
                    'disc_state_dict': disc.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'disc_optimizer': disc_optimizer.state_dict(),
                    'epoch': epoch + 1,
                }, os.path.join(args.output_dir, CHECKPOINT_FILE))

        torch.save(gen.state_dict(), os.path.join(args.output_dir, "generator_final.pth"))
        torch.save(disc.state_dict(), os.path.join(args.output_dir, "discriminator_final.pth"))

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Plotting the loss curves to analyze training progression
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_losses, label="Generator")
    plt.plot(disc_losses, label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(SAVE_IMAGES_PATH, "loss_plot.png"))
    plt.close()

    print("Generating t-SNE embeddings plot...")
    dataloader_for_embeddings, dataset_for_embeddings = get_dataloader(FIXED_IMAGE_SIZE, args.batch_size, args.data_root, shuffle=False)
    class_to_idx = dataset_for_embeddings.class_to_idx
    labels = []
    for _, label in dataloader_for_embeddings:
        labels.extend(label.tolist())
    plot_embeddings(dataloader_for_embeddings, labels, os.path.join(SAVE_IMAGES_PATH, "tsne_embeddings.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train StyleGAN2 on ADNI Dataset')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='Path to training data')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--output_dir', type=str, default=SAVE_MODEL_PATH, help='Directory to save training plots and models')
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    num_epochs = args.num_epochs
    BATCH_SIZE = args.batch_size
    SAVE_MODEL_PATH = args.output_dir
    SAVE_IMAGES_PATH = "./generated_images"

    train(args)
