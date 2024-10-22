import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Removing the classification layer

        # Adding a few fully connected layers to compute similarity
        self.fc1 = nn.Linear(resnet.fc.in_features * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # calculate similarity
        combined = torch.cat((output1, output2), 1)
        combined = func.relu(self.fc1(combined))
        combined = func.relu(self.fc2(combined))
        similarity = torch.sigmoid(self.fc3(combined))

        return similarity


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # distance between the output vectors
        euclidean_distance = func.pairwise_distance(output[0], output[1])

        # contrastive loss
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
