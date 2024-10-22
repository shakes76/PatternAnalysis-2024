import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
from tqdm import tqdm
import math

# Siamese Network Architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # First conv layer: 3 input channels, 64 output channels
            nn.ReLU(), # Non-linearity
            nn.MaxPool2d(2), # Reduce spatial dimensions
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Second conv layer: 64 input channels, 128 output channels
            nn.ReLU(), # Non-linearity
            nn.MaxPool2d(2) # Reduce spatial dimensions
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(128 * 64 * 64, 256), # Flatten and reduce to 256 dimensions
            nn.ReLU(), # Non-linearity
            nn.Linear(256, 128)  # Final embedding size of 128 for Euclidean distance calc
        )

    # Forward Pass
    def forward(self, x1, x2):
        # Process inputs through CNN
        out1 = self.cnn(x1)
        out2 = self.cnn(x2)

        # Flatten outputs
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)

        # Generate embeddings
        out1 = self.fc(out1)
        out2 = self.fc(out2)
        return out1, out2

# Contrastive Loss Function
def contrastive_loss(output1, output2, label, margin=1.0):
    # Euclidean distance between the outputs
    euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
    
    # Contrastive loss function
    # For similar pairs (label = 0), minimize distance
    # For dissimilar pairs (label = 1), push distance to be greater than margin
    loss = torch.mean((1 - label) * 0.5 * euclidean_distance ** 2 +
                      label * 0.5 * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

# Training loop with contrastive loss
def train_siamese_network(model, train_loader, epochs=5, margin=1.0):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Convert data to float and reshape
            img1, img2, labels = img1.float(), img2.float(), labels.unsqueeze(1).float()
            
            optimizer.zero_grad()  # Reset gradients
            output1, output2 = model(img1, img2)  # Forward pass
            loss = contrastive_loss(output1, output2, labels, margin=margin)  # Compute loss
            loss.backward()  # Backprop
            optimizer.step()  # Update weights

            running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")

            # Save checkpoint if it's the best model so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_model.pth')

            # Log epoch results
            with open('siamese_training.txt', 'a') as f:
                f.write(f"Epoch {epoch+1}, Loss: {epoch_loss}\n")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and optimizer
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam Optimizer

    # Train model
    train_siamese_network(model, train_loader, epochs=5, margin=1.0)
