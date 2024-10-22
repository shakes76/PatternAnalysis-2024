import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = CNN()
    
    def forward(self, input1, input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)

        distance = torch.abs(output1 - output2)
        
        similarity_score = torch.sum(distance, dim=1)
        
        return similarity_score

