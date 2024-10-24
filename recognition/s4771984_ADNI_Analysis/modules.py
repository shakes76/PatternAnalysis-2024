import torch
import torch.nn as nn
import warnings as w
w.filterwarnings('ignore')

class GFNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GFNet, self).__init__()
        
        # Initial convolutional layers with basic pooling
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2,2) # Reduce the dimensionality
        self.elu = nn.ELU() #Activation Function
        self.dropout = nn.Dropout(0.5)
        self.fc1 = None
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Apply the convolutional layers with ELU and Maxpool
       x = self.pool(self.elu(self.conv1(x)))
       x = self.pool(self.elu(self.conv2(x)))
       x = self.pool(self.elu(self.conv3(x)))
       
       # Flattening the output for the full connection of the layers
       x = x.view(x.size(0), -1)
       
       if self.fc1 is None:
           self.fc1 = nn.Linear(x.size(1), 256).to(x.device)
       
    # Now applying the fully connected layer with the dropout layer
       x = self.elu(self.fc1(x))
       x = self.dropout(x)
       x = self.fc2(x)

       return x
        
        