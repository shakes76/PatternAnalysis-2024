"""Model components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class SiameseNetwork(nn.Module):
    """A Siamese network with a ResNet backbone."""

    def __init__(self, pretrained=False):
        """
        Args:
            pretrained: Whether to use a ResNet with pretrained ImageNet weights.
        """
        super(SiameseNetwork, self).__init__()

        # Use a ResNet backbone without the last FC layer (we'll define our own)
        if pretrained:
            resnet_base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet_base = resnet50()
        self.net = nn.Sequential(
            *list(resnet_base.children())[:-1],
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64, bias=True)
        )

        # Freeze ResNet layers if we're using pretrained weights
        if pretrained:
            for param in self.net[:-3].parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward pass through the siamese network"""
        return self.net(x)

    def forward_pair(self, x1, x2):
        """Forward pass through the siamese network for a pair of inputs"""
        return self.net(x1), self.net(x2)


class MajorityClassifier:
    """
    A classifier that compares the similarity of unseen points to a set of reference
    images (via pairwise distance) and picks the class that the majority of reference
    images are similar to.
    """

    def __init__(self, margin=1.0):
        """
        Args:
            margin: The margin as in contrastive loss. Used to determine whether two
              images are considered a similar pair or not.
        """

        def threshold(x1, x2):
            dist = F.pairwise_distance(x1, x2)
            return (dist >= margin).float()

        self.threshold = threshold

    def fit(self, X, y):
        """
        Saves X and y as the reference embeddings and reference labels, respectively.
        """
        self.ref_embeddings = X
        self.ref_targets = y

    def predict_proba(self, X):
        """
        Returns the estimated probability of belonging to class 1 (i.e. malignant) for
        X, based on the reference set.
        """
        batch_size = X.shape[0]

        preds = []
        for ref_embedding, ref_target in zip(self.ref_embeddings, self.ref_targets):
            ref_embedding = ref_embedding.repeat(batch_size, 1)
            ref_target = ref_target.repeat(batch_size)

            pair_pred = self.threshold(ref_embedding, X)
            pred = torch.logical_xor(pair_pred, ref_target).float()
            preds.append(pred)

        # Stack predictions so that the dimensions are [batch_size, num_ref_imgs]
        preds = torch.stack(preds, dim=1)

        return preds.mean(dim=1)

    def predict(self, X):
        """
        Returns the estimated class for X ("hard" classification), based on the
        reference set.
        """
        return (self.predict_proba(X) >= 0.5).float()
