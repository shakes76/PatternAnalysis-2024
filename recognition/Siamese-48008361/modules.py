## Initial Comment

import torch
import torch.nn as nn
import torch.nn.functional as F

# Making our custom ResNet inspired by Resnet-18
# Residual block of ResNet
class ResBlock(nn.module):
    expansion = 1

    # Constructor for the ResBlock
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # Just two layers of Convolution and Batch Normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        # If the input and output channels are not the same, then we need to adjust the shortcut connection
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Making the ResNet class   
class CustomResNet(nn.module):
    
    def __init__(self, block, num_blocks, num_classes=2):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Initialising our ResNet
def ResNet18():
    return CustomResNet(ResBlock, [2, 2, 2, 2])


# Siamese Network Class
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        # Custom ResNet18 inspired network as the feature extractor
        resnet = ResNet18()
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
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
    


