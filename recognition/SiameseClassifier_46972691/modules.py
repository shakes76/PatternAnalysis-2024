# modules.py

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Embedding(nn.Module):
    def __init__(self):
        super(ResNet18Embedding, self).__init__()
        # Load ResNet18 model without pre-trained weights
        self.model = models.resnet18(weights=None)
        # Modify the last fully connected layer to output embeddings
        self.model.fc = nn.Identity()

    def forward(self, x):
        embeddings = self.model(x)
        return embeddings

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = ResNet18Embedding()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x1, x2):
        # Pass both images through the embedding network
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        # Compute the absolute difference between embeddings
        diff = torch.abs(out1 - out2)
        # Pass the difference through the fully connected layers
        out = self.fc(diff)
        return out  # Outputs raw logits

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.embedding_net = ResNet18Embedding()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        embeddings = self.embedding_net(x)
        out = self.fc(embeddings)
        return out  # Outputs raw logits
