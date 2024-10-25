# modules.py
# Contains the class defintions for the networks used
# author: Harrison Martin
import torch.nn as nn

class SimpleCNNEmbedding(nn.Module):
    def __init__(self):
        super(SimpleCNNEmbedding, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 256 x 256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 32 x 128 x 128

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 64 x 64 x 64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 128 x 32 x 32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: 256 x 1 x 1
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = SimpleCNNEmbedding()

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        return out1, out2  # Return embeddings directly

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.embedding_net = SimpleCNNEmbedding()
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        embeddings = self.embedding_net(x)
        out = self.fc(embeddings)
        return out  # Outputs raw logits
