import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        """
        Siamese Neural Network architecture based on the architecture defined in the paper
        "Siamese neural networks for one-shot image recognition" by
        G. Koch, R. Zemel, R. Salakhutdinov et al.
        """

        super(SiameseNetwork, self).__init__()

        # Define 4 convolutional layer + max pooling layer pairs.
        # The first input will be the 3 channeled RGB data of the image

        # Convolutional layers
        self.c1 = nn.Conv2d(3, 64, kernel_size=10)
        self.c2 = nn.Conv2d(64, 128, kernel_size=7)
        self.c3 = nn.Conv2d(128, 128, kernel_size=4)
        self.c4 = nn.Conv2d(128, 256, kernel_size=4)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(36864, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward_once(self, x):
        # Forward pass through convolutional layer + max pooling layer pairs
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        x = self.pool(F.relu(self.c3(x)))
        x = self.pool(F.relu(self.c4(x)))

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Forward pass through first fully connected layer
        x = F.relu(self.fc1(x))
        return x

    def forward(self, input1, input2):
        # Pass both images through the same SNN architecture
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute the L1 distance between the resulting outputs
        l1_dist = torch.abs(output1 - output2)

        # Pass through the final fully connected layer
        output = torch.sigmoid(self.fc2(l1_dist))
        return output
