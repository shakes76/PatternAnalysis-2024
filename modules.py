import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.down1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.down4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.compress3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.compress2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.compress1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        CHANNEL_DIM = -3
        x1 = F.relu(self.max_pool(self.down1(x)))
        x2 = F.relu(self.max_pool(self.down2(x1)))
        x3 = F.relu(self.max_pool(self.down3(x2)))
        x4 = F.relu(self.max_pool(self.down4(x3)))
        x4_up = F.relu(self.up4(x4))
        x3_up = F.relu(self.up3(F.relu(self.compress3(torch.concat((x4_up, x3), dim=CHANNEL_DIM)))))
        x2_up = F.relu(self.up2(F.relu(self.compress2(torch.concat((x3_up, x2), dim=CHANNEL_DIM)))))
        x1_up = F.sigmoid(self.up1(F.relu(self.compress1(torch.concat((x2_up, x1), dim=CHANNEL_DIM)))))
        return x1_up

def f1_score(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    dims = (-3, -2, -1)
    EPSILON = 1e-8
    precision = torch.sum(output * target, dims) / (torch.sum(output, dims) + 1e-8)
    precision = (torch.abs(torch.sum(output, dims)) >= EPSILON) * precision
    recall = torch.sum(output * target, dims) / (torch.sum(output * target, dims) + torch.sum((1 - output) * target, dims) + 1e-8)
    recall = (torch.abs(torch.sum(output * target, dims) + torch.sum((1 - output) * target, dims)) >= EPSILON) * recall

    f1 = 1 - 2 * precision * recall / (precision + recall + 1e-8)
    f1 = (torch.abs(precision) >= EPSILON) * (torch.abs(recall) >= EPSILON) * f1
    return f1
