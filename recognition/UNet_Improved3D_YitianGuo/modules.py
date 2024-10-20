import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MRIDataset
from dataset import train_transforms, val_transforms
import numpy as np

from torch.utils.data import Dataset, DataLoader
in_chanel = 1 # grey
out_chanel = 6 # background, body outline, bone, bladder, rectum, prostate

class DoubleConv(nn.Module):
    """包含残差连接的两个连续卷积层"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.relu = nn.LeakyReLU(inplace=True)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    """注意力门"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class DownSample(nn.Module):
    """下采样层，通过 MaxPool3d 减少空间尺寸"""

    def __init__(self, kernel_size=2, stride=2):
        super(DownSample, self).__init__()
        self.down = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    """上采样层，包含注意力机制"""

    def __init__(self, in_channels, out_channels, attention_channels, kernel_size=2, stride=2):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.attention = AttentionBlock(F_g=out_channels, F_l=attention_channels, F_int=out_channels // 2)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        # 应用注意力机制
        skip = self.attention(g=x, x=skip)
        x = torch.cat((x, skip), dim=1)
        return x


class UNet3D(nn.Module):
    """3D UNet 模型"""

    def __init__(self, in_channels=in_chanel, out_channels=out_chanel, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        # 编码器部分
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.pool1 = DownSample()

        self.encoder2 = DoubleConv(features[0], features[1])
        self.pool2 = DownSample()

        self.encoder3 = DoubleConv(features[1], features[2])
        self.pool3 = DownSample()

        self.encoder4 = DoubleConv(features[2], features[3])
        self.pool4 = DownSample()

        # 瓶颈部分
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # 解码器部分，添加 attention_channels 参数
        self.upconv4 = UpSample(features[3] * 2, features[3], attention_channels=features[3])
        self.decoder4 = DoubleConv(features[3] * 2, features[3])

        self.upconv3 = UpSample(features[3], features[2], attention_channels=features[2])
        self.decoder3 = DoubleConv(features[2] * 2, features[2])

        self.upconv2 = UpSample(features[2], features[1], attention_channels=features[1])
        self.decoder2 = DoubleConv(features[1] * 2, features[1])

        self.upconv1 = UpSample(features[1], features[0], attention_channels=features[0])
        self.decoder1 = DoubleConv(features[0] * 2, features[0])

        # 最终输出层保持不变
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        # 移除 Softmax
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)  # [B, 16, D, H, W]
        enc2 = self.encoder2(self.pool1(enc1))  # [B, 32, D/2, H/2, W/2]
        enc3 = self.encoder3(self.pool2(enc2))  # [B, 64, D/4, H/4, W/4]
        enc4 = self.encoder4(self.pool3(enc3))  # [B, 128, D/8, H/8, W/8]

        bottleneck = self.bottleneck(self.pool4(enc4))  # [B, 256, D/16, H/16, W/16]

        # 解码器
        dec4 = self.upconv4(bottleneck, enc4)  # [B, 128, D/8, H/8, W/8]
        dec4 = self.decoder4(dec4)  # [B, 128, D/8, H/8, W/8]

        dec3 = self.upconv3(dec4, enc3)  # [B, 64, D/4, H/4, W/4]
        dec3 = self.decoder3(dec3)  # [B, 64, D/4, H/4, W/4]

        dec2 = self.upconv2(dec3, enc2)  # [B, 32, D/2, H/2, W/2]
        dec2 = self.decoder2(dec2)  # [B, 32, D/2, H/2, W/2]

        dec1 = self.upconv1(dec2, enc1)  # [B, 16, D, H, W]
        dec1 = self.decoder1(dec1)  # [B, 16, D, H, W]

        out = self.final_conv(dec1)  # [B, out_channels, D, H, W]
        # 移除 Softmax
        # out = self.softmax(out)
        return out

if __name__ == '__main__':
    # 指定图像和标签的文件夹路径
    train_image_folder = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon"
    train_label_folder = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon"

    # 获取所有图像和标签路径
    train_image_paths = sorted(glob.glob(os.path.join(train_image_folder, "*.nii*")))
    train_label_paths = sorted(glob.glob(os.path.join(train_label_folder, "*.nii*")))

    # 打印一些路径来验证（可选）
    print("Image paths:", train_image_paths[:3])  # 打印前三个路径，检查路径是否正确
    print("Label paths:", train_label_paths[:3])  # 打印前三个路径，检查路径是否正确

    # 构建数据集
    train_dataset = MRIDataset(
        image_paths=train_image_paths,
        label_paths=train_label_paths,
        transform=train_transforms,
        norm_image=True,
        dtype=np.float32
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    model = UNet3D()
    # 验证 DataLoader 和 Transform 是否正确工作
    for batch in train_loader:
        images, labels = batch['image'], batch['label']
        # 确保图像形状是符合模型的输入要求
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")

        output = model(images)
        print('output.shape:', output.shape)
        break