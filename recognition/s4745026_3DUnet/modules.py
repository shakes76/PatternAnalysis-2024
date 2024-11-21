import torch
import torch.nn as nn


class Basic3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, init_features=32):
        super(Basic3DUNet, self).__init__()

        features = init_features

        # Encoder
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(
            features * 8, features * 16, name="bottleneck")

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(
            (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(
            (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(
            (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # Final Convolution
        self.conv = nn.Conv3d(in_channels=features,
                              out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=features, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True)
        )


class Improved3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(Improved3DUNet, self).__init__()

        features = init_features

        # Encoder
        self.encoder1 = self._residual_block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._residual_block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._residual_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = self._residual_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._residual_block(features * 8, features * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._residual_block(features * 8 * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._residual_block(features * 4 * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._residual_block(features * 2 * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._residual_block(features * 2, features)

        # Final Convolution
        self.conv = nn.Conv3d(in_channels=features,
                              out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)

    def _residual_block(self, in_channels, features):
        return ResidualBlock(in_channels, features)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, features,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            features, features, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(features)

        # Identity layer for residual connection
        self.identity = nn.Conv3d(
            in_channels, features, kernel_size=1, bias=False) if in_channels != features else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # If identity layer is defined, transform the input
        if self.identity is not None:
            identity = self.identity(x)

        out += identity
        out = self.relu(out)
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None

    def forward(self, y_pred, y_true):
        # Check shape sizes match
        assert y_pred.size() == y_true.size(
        ), f"Shape mismatch: {y_pred.size()} != {y_true.size()}"

        batch_size = y_pred.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.contiguous().view(batch_size, num_classes, -1)
        y_true = y_true.contiguous().view(batch_size, num_classes, -1)

        # DSC Calculation
        intersection = (y_pred * y_true).sum(2)
        union = y_pred.sum(2) + y_true.sum(2)
        dsc = (2. * intersection + self.smooth) / (union + self.smooth)

        if self.weights is not None:
            if len(self.weights) != num_classes:
                raise ValueError(
                    f"Expected weights of length {num_classes}, but got {len(self.weights)}")
            dsc = dsc * self.weights.to(y_pred.device)

        return 1. - dsc.mean()


def calculate_dice(y_pred, y_true, num_classes=6):
    assert y_true.size() == y_pred.size(
    ), f"Shape mismatch: {y_pred.size()} != {y_true.size()}"

    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)

    dice_scores = []
    for cls in range(num_classes):
        y_pred_cls = (y_pred == cls).float()
        y_true_cls = (y_true == cls).float()

        intersection = (y_pred_cls * y_true_cls).sum(dim=[1, 2, 3])
        union = y_pred_cls.sum(dim=[1, 2, 3]) + y_true_cls.sum(dim=[1, 2, 3])
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_scores.append(dice.mean().item())

    return torch.tensor(dice_scores, device=y_pred.device)
