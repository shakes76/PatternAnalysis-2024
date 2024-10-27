import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MRIDataset_pelvis
from torch.utils.data import DataLoader


class UNet3D(nn.Module):
    """3D U-Net model for segmentation tasks."""

    def __init__(self, in_channel=1, out_channel=6):
        super(UNet3D, self).__init__()

        # Encoder layers with Batch Normalization
        self.encoder1 = self.conv_block(in_channel, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)

        # Decoder layers with Dropout
        self.decoder2 = self.deconv_block(256, 128)
        self.decoder3 = self.deconv_block(128, 64)
        self.decoder4 = self.deconv_block(64, 32)
        self.decoder5 = nn.Conv3d(32, out_channel, kernel_size=1)  # Output layer

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # Encoder part
        out1 = self.encoder1(x)
        out2 = self.encoder2(F.max_pool3d(out1, kernel_size=2, stride=2))
        out3 = self.encoder3(F.max_pool3d(out2, kernel_size=2, stride=2))
        out4 = self.encoder4(F.max_pool3d(out3, kernel_size=2, stride=2))

        # Decoder part
        out = F.interpolate(self.decoder2(out4), scale_factor=(2, 2, 2), mode='trilinear')
        out = out + out3

        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear')
        out = out + out2

        out = F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear')
        out = out + out1

        out = self.decoder5(out)  # No activation here, will apply softmax during evaluation
        return out


class DiceLoss(nn.Module):
    """Dice loss for evaluating segmentation accuracy."""

    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, f"Shapes don't match: {inputs.shape} != {targets.shape}"
        inputs = inputs[:, 1:]  # Skip background class
        targets = targets[:, 1:]  # Skip background class
        axes = tuple(range(2, len(inputs.shape)))  # Sum over elements per sample and per class
        intersection = torch.sum(inputs * targets, axes)
        addition = torch.sum(inputs ** 2 + targets ** 2, axes)
        return 1 - torch.mean((2 * intersection + self.smooth) / (addition + self.smooth))


if __name__ == '__main__':
    # Testing the dataset and model
    test_dataset = MRIDataset_pelvis(mode='test', dataset_path=r"path_to_your_dataset")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
    model = UNet3D(in_channel=1, out_channel=6)

    for batch_ndx, sample in enumerate(test_dataloader):
        print('Test batch:')
        print('Image shape:', sample[0].shape)
        print('Label shape:', sample[1].shape)
        output = model(sample[0])
        print('Output shape:', output.shape)

        labely = torch.nn.functional.one_hot(sample[1].squeeze(1).long(), num_classes=6).permute(0, 4, 1, 2, 3).float()
        break
