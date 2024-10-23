import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import VQVAE  # import VQVAE module
from dataset import get_data_loader  # import data loader
from torchvision import models
import torch.nn as nn

batch_size = 32
learning_rate = 1e-4
num_epochs = 50
perceptual_loss_weight = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = '/content/data/HipMRI_study_keras_slices_data'
train_loader = get_data_loader(root_dir=data_dir, subset='train', batch_size=batch_size, target_size=(256, 256))

model = VQVAE(in_channels=1, hidden_channels=256, num_embeddings=1024, embedding_dim=128).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

vgg = models.vgg16(pretrained=True).features[:16].to(device)
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def preprocess_vgg(x):
    # single channel -> three channels
    x = x.repeat(1, 3, 1, 1)
    # Normalization
    x = (x - imagenet_mean) / imagenet_std
    return x

def perceptual_loss(x_recon, x):
    features_recon = vgg(x_recon)
    features_x = vgg(x)
    return torch.nn.functional.l1_loss(features_recon, features_x)

# training model
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        # Forward Propagation
        reconstructed, vq_loss = model(batch)
        recon_loss = torch.nn.functional.l1_loss(reconstructed, batch)

        # Calculating Perceptual Loss
        x_recon_vgg = preprocess_vgg(reconstructed)
        x_vgg = preprocess_vgg(batch)
        p_loss = perceptual_loss(x_recon_vgg, x_vgg)

        loss = recon_loss + perceptual_loss_weight * p_loss + vq_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    # Learning rate scheduling
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# save model
torch.save(model.state_dict(), 'vqvae_hipmri.pth')