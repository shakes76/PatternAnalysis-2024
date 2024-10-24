import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

def PCA_transform(input_data, n_components, shape):
    mean = torch.mean(input_data, dim=0)
    X -= mean

    U, S, V = torch.linalg.svd(X, full_matrices=False)
    components = V[:n_components]

    transformed_data = components.view(n_components, shape[0], shape[1])

    return transformed_data

class CNN(nn.Module):
    def __init__(self, shape, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.ModuleList([None] * 4)
        self.conv[0] = nn.Conv2d(3, 64, kernel_size=10)
        self.conv[1] = nn.Conv2d(64, 128, kernel_size=7)
        self.conv[2] = nn.Conv2d(128, 128, kernel_size=4)
        self.conv[3] = nn.Conv2d(128, 128, kernel_size=4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self._get_conv_output_size(shape), num_classes)

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, 3, *shape)
        for conv in self.conv:
            x = self.pool(self.relu(conv(x)))
        return x.numel()

    def forward(self, x):
        for conv in self.conv:
            x = self.pool(self.relu(conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(SiameseNetwork, self).__init__()
        self.resnet = resnet50(pretrained=True)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)

        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive