import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import VQVAE  # import VQVAE module
from dataset import get_data_loader  # import data loader

batch_size = 16
learning_rate = 1e-3
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = '/content/data/HipMRI_study_keras_slices_data'
train_loader = get_data_loader(root_dir=data_dir, subset='train', batch_size=batch_size, target_size=(256, 256))


model = VQVAE(in_channels=1, hidden_channels=128, num_embeddings=512, embedding_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training model
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        # forward
        reconstructed, vq_loss = model(batch)
        recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)
        loss = recon_loss + vq_loss

        # back propagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")