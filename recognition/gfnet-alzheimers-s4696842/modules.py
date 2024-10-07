import torch
import torch.nn as nn
import torch.fft


class GlobalFilterLayer(nn.Module):
    def __init__(self, in_channels, height, width):
        super(GlobalFilterLayer, self).__init__()

        self.filter_real = nn.Parameter(torch.randn(in_channels, height, width))
        self.filter_imag = nn.Parameter(torch.randn(in_channels, height, width))

    def forward(self, x):
        # Convert to frequency domain.
        x_fft = torch.fft.fft2(x)

        real_part, imag_part = x_fft.real, x_fft.imag

        # Apply filtering.
        filtered_real = real_part * self.filter_real - imag_part * self.filter_imag
        filtered_imag = real_part * self.filter_imag + imag_part * self.filter_real

        # Apply inverse FFT to get back to spatial domain.
        filtered_x_fft = torch.complex(filtered_real, filtered_imag)
        x_filtered = torch.fft.ifft2(filtered_x_fft)

        # Return real part as the output since input is real.
        return x_filtered.real


class GFNetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, height, width, kernel_size=3, padding=1
    ):
        super(GFNetBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.global_filter = GlobalFilterLayer(out_channels, height, width)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.global_filter(x)
        return x


class GFNet(nn.Module):
    def __init__(self, in_channels, num_classes, height, width):
        super(GFNet, self).__init__()
        self.layer1 = GFNetBlock(in_channels, 64, height, width)
        self.layer2 = GFNetBlock(64, 128, height, width)
        self.layer3 = GFNetBlock(128, 256, height, width)
        self.fc = nn.Linear(256 * height * width, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
