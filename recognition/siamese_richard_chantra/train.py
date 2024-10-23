import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DataManager
from tqdm import tqdm
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class SiameseNetwork(nn.Module):
    """
    Siamese Network for learning image embeddings of benign and malignant melanomas.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # ResNet50 Feature Extractor
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),  # Output of ResNet50 is a 2048 dim feature vector
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)    # Final embedding size of 128 for Euclidean distance calc
        )

    def forward(self, x1, x2):
        """Forward pass to compute embeddings for a pair of images"""
        # Get embeddings for both images
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        return out1, out2
    
    def get_embedding(self, x):
        """Computing embeddings for a single image"""
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

class MLPClassifier(nn.Module):
    """
    MLP Classifier using Siamese Network embeddings to predict melanoma
    """
    def __init__(self, embedding_dim=128):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding):
        """
        Input: embedding from Siamese network
        Output: probability of being malignant (0 = benign, 1 = malignant)
        """
        return self.classifier(embedding)

def contrastive_loss(output1, output2, label, margin=1.0):
    """
    Contrastive loss for Siamese Network training
    """
    # Calculate euclidean distance
    euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-6)
    
    # Calculate contrastive loss
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                     label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    
    return loss

def train_siamese_network(siamese_network, optimizer, train_loader, epochs=5, margin=1.0):
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
            
            # Forward pass
            optimizer.zero_grad()
            embedding1, embedding2 = siamese_network(img1, img2)
            
            # Calculate loss
            loss = contrastive_loss(embedding1, embedding2, similarity_label, margin)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(siamese_network.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Siamese Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': siamese_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_siamese_network.pth')

def train_mlp_classifier(siamese_network, mlp_classifier, optimizer, train_loader, epochs=5):
    """
    Train MLP classifier using Siamese Network embeddings
    """
    mlp_classifier.train()
    siamese_network.eval()  # Freeze Siamese network
    criterion = nn.BCELoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Classifier"):
            # Get image and its label
            img1 = batch['img1'].to(device)
            diagnosis_label = batch['diagnosis1'].to(device).unsqueeze(1)
            
            # Get embeddings without gradient tracking
            with torch.no_grad():
                embeddings = siamese_network.get_embedding(img1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = mlp_classifier(embeddings)
            loss = criterion(outputs, diagnosis_label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += diagnosis_label.size(0)
            correct += (predicted == diagnosis_label).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Classifier Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': mlp_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'best_mlp_classifier.pth')

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SKIP_SIAMESE_TRAINING = 1  # 0: Begin training, 1: Skip training (use checkpoint)

    # Setup data
    data_manager = DataManager('archive/train-metadata.csv', 'archive/train-image/image/')
    data_manager.load_data()
    data_manager.create_dataloaders()
    train_loader = data_manager.train_loader
    test_loader = data_manager.test_loader

    # Initialize models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Initialize optimizers
    optimizer_siamese = optim.Adam(siamese_network.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_mlp = optim.Adam(mlp_classifier.parameters(), lr=0.001, weight_decay=1e-4)

    if SKIP_SIAMESE_TRAINING:
        # Load Siamese Network checkpoint
        print("Loading Siamese Network checkpoint...")
        checkpoint = torch.load('best_siamese_network.pth')
        siamese_network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded Siamese Network checkpoint with loss: {checkpoint['loss']:.4f}")
    else:
        # Train Siamese network from scratch
        print("Training Siamese Network to learn embeddings from images:")
        train_siamese_network(siamese_network, optimizer_siamese, train_loader, epochs=10)
    
    # Train classifier
    print("\nTraining MLPClassifier using learned embeddings:")
    train_mlp_classifier(siamese_network, mlp_classifier, optimizer_mlp, train_loader, epochs=10)