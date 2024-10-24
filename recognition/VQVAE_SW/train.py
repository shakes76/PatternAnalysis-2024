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

# Weight assigned to the perceptual loss in the total loss function
perceptual_loss_weight = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = '/content/data/HipMRI_study_keras_slices_data'
train_loader = get_data_loader(root_dir=data_dir, subset='train', batch_size=batch_size, target_size=(256, 256))

# Initialize the VQVAE model
model = VQVAE(in_channels=1, hidden_channels=256, num_embeddings=1024, embedding_dim=128).to(device)

# Define the optimizer as Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define a learning rate scheduler that reduces the learning rate by a factor of 0.5 every 20 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Load the pre-trained VGG16 model from torchvision and use its features up to layer 16 for perceptual loss
vgg = models.vgg16(pretrained=True).features[:16].to(device)
vgg.eval() # Set the VGG model to evaluation mode to ensure the pre-trained weights are not updated

# Freeze the VGG model parameters to prevent them from being updated during training
for param in vgg.parameters():
    param.requires_grad = False

# Define ImageNet mean and standard deviation for preprocessing images for VGG
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def preprocess_vgg(x):
    """
    Function to preprocess input images for VGG.
    Convert single-channel images to three-channel images and normalize using ImageNet stats.
    """
    # match the input requirements of VGG
    x = x.repeat(1, 3, 1, 1)
    # Normalization
    x = (x - imagenet_mean) / imagenet_std
    return x

def perceptual_loss(x_recon, x):
    """
    Calculate perceptual loss between the reconstructed image and the original image.
    Extract feature maps for both the reconstructed and original images using the VGG model.
    """
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

        # Calculate L1 loss (mean absolute error) between the original image and the reconstructed image
        recon_loss = torch.nn.functional.l1_loss(reconstructed, batch)

        # Calculating Perceptual Loss
        x_recon_vgg = preprocess_vgg(reconstructed)
        x_vgg = preprocess_vgg(batch)
        p_loss = perceptual_loss(x_recon_vgg, x_vgg)

        # Total loss
        loss = recon_loss + perceptual_loss_weight * p_loss + vq_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    # Update the learning rate using the scheduler after every epoch
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# save model
torch.save(model.state_dict(), 'vqvae_hipmri.pth')