import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
from tqdm import tqdm
import math
import torchvision.models as models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Siamese Network Architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # ResNet50 Feature Extractor
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),  # ResNet50 outputs 2048-dim feature vectors
            nn.ReLU(),  # Non-linearity
            nn.Linear(256, 128)  # Final embedding size of 128 for Euclidean distance calc
        )

    # Forward Pass
    def forward(self, x1, x2):
        # Process inputs through ResNet50
        out1 = self.features(x1)
        out2 = self.features(x2)

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
            img1, img2 = img1.to(device), img2.to(device) # Move tensors to device
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Reset gradients
            output1, output2 = model(img1, img2)  # Forward pass
            loss = contrastive_loss(output1, output2, labels, margin=margin)  # Compute loss
            loss.backward()  # Backprop
            optimizer.step()  # Update weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")

        # Save checkpoint for best model
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

if __name__ == "__main__":
    # Initialize model and move to device
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam Optimizer

    # Train model
    train_siamese_network(model, train_loader, epochs=5, margin=1.0)