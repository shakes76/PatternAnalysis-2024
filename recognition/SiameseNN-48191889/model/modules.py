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

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer sizes will be initialized dynamically
        self.fc1 = None
        self.fc2 = nn.Linear(4096, 1)

    def forward_once(self, x):
        # Forward pass through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Dynamically create and move fc1 to the correct device
        if self.fc1 is None:
            flattened_size = x.size(1)
            self.fc1 = nn.Linear(flattened_size, 4096).to(x.device)

        # Forward pass through the fully connected layer
        x = F.relu(self.fc1(x))
        return x

    def forward(self, input1, input2):
        # Forward pass through both branches
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute the L1 distance between the two outputs
        l1_distance = torch.abs(output1 - output2)

        # Pass through the final fully connected layer
        output = torch.sigmoid(self.fc2(l1_distance))
        return output
