import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F


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


class Classifier:
    def __init__(self, margin=0.5):
        self.margin = margin
        self.reference_set = []

    def set_reference_set(self, reference_embeddings, reference_labels):
        """
        Set the reference set using a batch of embeddings and their corresponding labels.
        :param reference_embeddings: Tensor of shape (N, D) where N is the number of reference samples and D is the embedding size.
        :param reference_labels: Tensor of shape (N,) where N is the number of reference samples.
        """
        self.reference_set = [(embedding, label) for embedding, label in zip(reference_embeddings, reference_labels)]

    def predict_class(self, embedding):
        """
        Predict the class of the input embedding based on the margin.
        :param embedding: Tensor of shape (D,)
        :return: Predicted class (0 or 1).
        """
        positive_distances = []
        negative_distances = []

        for ref_embedding, label in self.reference_set:
            distance = F.pairwise_distance(embedding.unsqueeze(0), ref_embedding.unsqueeze(0), p=2).item()
            if label == 1:
                positive_distances.append(distance)
            else:
                negative_distances.append(distance)

        # Calculate average distances to positive and negative samples
        avg_positive_distance = sum(positive_distances) / len(positive_distances) if positive_distances else float('inf')
        avg_negative_distance = sum(negative_distances) / len(negative_distances) if negative_distances else float('inf')

        # Predict class based on the margin
        return 1 if avg_positive_distance < avg_negative_distance - self.margin else 0
