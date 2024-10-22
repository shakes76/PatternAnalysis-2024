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

# Original Siamese Network (unchanged)
class SiameseNetwork(nn.Module):
    """
    Siamese Network for learning image embeddings of benign and malignant melanomas.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # ResNet50 Feature Extractor
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(2048, 256), # Output of ResNet50 is a 2048 dim feature vector
            nn.ReLU(),
            nn.Linear(256, 128) # Final embedding size of 128 for Euclidean distance calc
        )

    def forward(self, x1, x2):
        """
        The forward pass to compute embeddings for a pair of images
        """
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
    
    def get_embedding(self, x):
        """
        Computing the embeddings for a single image
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding_network(features)
        return embedding

# Separate MLP Classifier
class MLPClassifier(nn.Module):
    """
    Using a Multi-layer Perceptron using Siamese Network embeddings to predict malignant melanoma
    """
    def __init__(self, embedding_dim=128):
        super(MelanomaClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, embedding):
        """
        Input: embedding from Siamese network
        Output: probability of image being malignant (0 = benign, 1 = malignant)
        """
        return self.classifier(embedding)

# Contrastive Loss Function
def contrastive_loss(output1, output2, label, margin=1.0):
    """
    Contrastive loss used in the training of Siamese Network
    """
    euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
    loss = torch.mean((1 - label) * 0.5 * euclidean_distance ** 2 +
                      label * 0.5 * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

# Training Siamese Network
def train_siamese_network(siamese_network, train_loader, epochs=5, margin=1.0):
    """
    Train Siamese Network to learn embeddings from images
    """
    siamese_network.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Siamese"):
            # Get batch data
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            similarity_label = batch['similarity_label'].to(device)
            
            # Compute embeddings and loss
            siamese_network_optimizer.zero_grad()
            embedding1, embedding2 = model(img1, img2)
            loss = contrastive_loss(embedding1, embedding2, similarity_label, margin)
            loss.backward()
            siamese_network_optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Siamese Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")

        # Save checkpoint for the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': siamese_network.state_dict(),
                'optimizer_state_dict': siamese_network_optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_siamese_network.pth')

# Train MLP Classifier using Siamese Network embeddings
def train_mlp_classifier(siamese_network, mlp_classifier, train_loader, epochs=5):
    """
    Train MLP classifier to diagnose melanoma using learned embeddings
    """
    classifier.train()
    criterion = nn.BCELoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Classifier"):
            # Get first image and its diagnosis from batch
            img1 = batch['img1'].to(device)
            diagnosis_label = batch['diagnosis1'].to(device)  # 0 = benign, 1 = malignant
            
            # Get embeddings from Siamese Network
            with torch.no_grad():
                embeddings = siamese_network.get_embedding(img1)
            
            # Classify embeddings
            optimizer_classifier.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer_classifier.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += binary_labels.size(0)
            correct += (predicted == binary_labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Classifier Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer_classifier.state_dict(),
                'accuracy': best_acc,
            }, 'best_mlp_classifier.pth')

if __name__ == "__main__":
    # Initialize models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Initialize optimizers
    optimizer_siamese_network = optim.Adam(siamese_network.parameters(), lr=0.001)
    optimizer_mlp_classifier = optim.Adam(mlp_classifier.parameters(), lr=0.001)

    # First train Siamese network
    print("Training Siamese Network to learn embeddings from images:")
    train_siamese_network(siamese_network, train_loader, epochs=5)
    
    # Then train classifier
    print("\nTraining MLPClassifier using learned embeddings:")
    siamese_network.eval() # Using the embeddings only
    train_classifier(siamese_network, mlp_classifier, train_loader, epochs=5)