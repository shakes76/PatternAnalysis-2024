import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# CNN model for image classification, used later for siamese network
class CNN(nn.Module):
    def __init__(self, shape, num_classes):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output_size(shape), 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes)
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

# Siamese Network for learning similarity between two images, consisting of two identical CNNs
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = CNN((256,256), 2)
    
    def forward(self, input1, input2):
        output1 = self.cnn.forward(input1)
        output2 = self.cnn.forward(input2)

        return output1, output2

# Loss function for the Siamese Network
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive