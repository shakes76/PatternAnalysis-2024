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
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output_size(shape), 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, num_classes)
        )

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, 3, *shape)
        x = self.cnn(x)
        x = self.flatten(x)
        return x.numel()

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = CNN((256,256), 2)
    
    def forward(self, input1, input2):
        output1 = self.cnn.forward(input1)
        output2 = self.cnn.forward(input2)

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