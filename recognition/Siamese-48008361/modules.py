## Initial Comment

import torch
import torch.nn as nn
import torch.nn.functional as F



# Siamese Network Class
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=64):
        super(SiameseNetwork, self).__init__()

        # Convolutional Neural Network Architecture

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Fully connected layer to get the embeddings
        self.fc = nn.Linear(512, embedding_dim)
        # Classifier to classify the embeddings
        self.classifier = nn.Linear(embedding_dim, 2)
    
    def forward_once(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    # Forward pass for the Siamese Network (we are passing in anchor, positive and negative images)

    def forward(self, x1, x2, x3):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out3 = self.forward_once(x3)
        return out1, out2, out3
    
    def get_embedding(self, x):
        return self.forward_once(x)
    
    def classify(self, x):
        embedding = self.get_embedding(x)
        return self.classifier(embedding)


# Triplet Loss Function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()
    

# Basic helpers to later be imported in train.py
def get_model(embedding_dim=64):
    return SiameseNetwork(embedding_dim=embedding_dim)

def get_loss(margin=1.0):
    return TripletLoss(margin)