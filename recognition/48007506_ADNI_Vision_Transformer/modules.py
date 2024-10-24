"""
modules.py

Source code of the components of the vision transformer.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
import torch.nn as nn
from torchvision import models
from constants import DROPOUT_RATE

class GlobalFilterLayer(nn.Module):
    """
    Layer that applies a learnable global filter to the input using Fourier Transform.
    """
    def __init__(self, in_channels, height, width):
        super(GlobalFilterLayer, self).__init__()

        # Layer normalization to stabilise the training
        self.layer_norm = nn.LayerNorm(normalized_shape=(in_channels, height, width))
        
        # Learnable filter initialised with random values
        self.learnable_filter = nn.Parameter(torch.randn(in_channels, height, width, dtype=torch.float32))

    def forward(self, x):
        x = self.layer_norm(x)                              # Normalize the input tensor
        x_fft = torch.fft.fft2(x)                           # Apply 2D Fast Fourier Transform (FFT)
        filter_fft = torch.fft.fft2(self.learnable_filter)  # Apply FFT to the learnable filter
        x_fft_filtered = x_fft * filter_fft                 # Element-wise multiplication of the input FFT and the filter FFT
        x = torch.fft.ifft2(x_fft_filtered)                 # Apply 2D Inverse Fast Fourier Transform (IFFT) to recover the filtered spatial domain
        return x.abs()                                      # Return the magnitude of the resulting tensor after IFFT

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) that consists of linear layers, batch normalization,
    activation functions, and dropout for regularization.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        # Build MLP layers dynamically based on provided hidden dimensions
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))  # Add a linear layer
            layers.append(nn.BatchNorm1d(hidden_dim))        # Add batch normalization
            layers.append(nn.GELU())                         # Add GELU activation function
            layers.append(nn.Dropout(dropout_rate))          # Add dropout for regularisation
            input_dim = hidden_dim                           # Update input dimension for the next layer
        layers.append(nn.Linear(hidden_dim, output_dim))     # Final output layer
        self.model = nn.Sequential(*layers)                  # Sequential model from the list of layers

    def forward(self, x):
        return self.model(x)  # Pass input through the MLP layers

class GFNet(nn.Module):
    """
    Main class for GFNet model, which incorporates a pretrained ResNet backbone,
    a global filter layer, and a feed-forward network for classification.
    """
    def __init__(self, num_classes=2):
        super(GFNet, self).__init__()
        
        # Load a pretrained ResNet-18 model for feature extraction and patch embedding
        resnet = models.resnet18(pretrained=True)

        # Remove the classification head but keep the feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Global Filter Layer (Learnable Filter in Frequency Domain)
        self.global_filter = GlobalFilterLayer(in_channels=512, height=4, width=4)

        # Feed Forward Network (LayerNorm + MLP)
        self.ffn_norm = nn.LayerNorm(512 * 4 * 4)
        self.ffn_mlp = MLP(512 * 4 * 4, hidden_dims=[1024, 512, 256], output_dim=512 * 4 * 4, dropout_rate=DROPOUT_RATE)

        # Global Average Pooling to reduce dimensionality
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Classifier layer to output final class probabilities
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Patch Embedding using ResNet Backbone
        x = self.backbone(x)

        # Apply the Global Filter Layer
        x = self.global_filter(x)

        # Flatten the output for Feed Forward Network
        x = x.view(x.size(0), -1)

        # Pass through the Feed Forward Network (LayerNorm + MLP)
        x = self.ffn_norm(x)
        x = self.ffn_mlp(x)

        # Reshape for Global Average Pooling
        x = x.view(x.size(0), 512, 4, 4)
        x = self.global_avg_pooling(x)

        # Flatten the pooled output for the classifier
        x = x.view(x.size(0), -1)

        # Classify the final output
        x = self.classifier(x)

        return x

