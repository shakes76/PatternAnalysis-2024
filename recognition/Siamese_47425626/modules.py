import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load pretrained ResNet-50 models for feature extraction
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the classification layer to get feature embeddings

        # Add a small network after the feature extraction for contrastive loss
        self.embedding_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        # Define a transform to ensure input is of the correct type (float)
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
        ])

    def forward(self, x):
        # Apply transformation to ensure input type is float
        x = self.transform(x)
        # Extract features using ResNet-50
        x = self.feature_extractor(x)
        # Pass through the embedding layer to get final embedding
        x = self.embedding_layer(x)
        return x