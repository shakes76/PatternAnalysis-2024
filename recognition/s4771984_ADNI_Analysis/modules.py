import torch
import torch.nn as nn
import warning as w
w.filterwarnings('ignore')

class GFNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GFNet, self).__init__()
        # Convolutional layers with 96 and 192 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)

        # Global Filter Layer
        self.global_filter = nn.Conv2d(192, 192, kernel_size=1)

        # Fully connected layers
        self.fc1 = None  # Placeholder for fc1, initialized in forward based on input size
        self.fc2 = nn.Linear(256, num_classes)  # Output layer

        # Pooling, dropout, and activation layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.elu = nn.ELU()  # ELU activation function

    def forward(self, x):
        x = self.pool(self.elu(self.conv1(x)))
        x = self.pool(self.elu(self.conv2(x)))
        x = self.pool(self.elu(self.conv3(x)))
        x = self.pool(self.elu(self.conv4(x)))
        x = self.global_filter(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)
        
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x