import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder (Reduced depth)
        self.conv1 = conv_block(1, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)

        # Bottleneck (Remove)
        self.bottleneck = conv_block(256, 512)

        # Decoder (Reduced depth)
        self.upconv3 = up_block(512, 256)
        self.conv3_1 = conv_block(512, 256)

        self.upconv2 = up_block(256, 128)
        self.conv2_1 = conv_block(256, 128)

        self.upconv1 = up_block(128, 64)
        self.conv1_1 = conv_block(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, 6, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(nn.MaxPool2d(2)(conv1))
        conv3 = self.conv3(nn.MaxPool2d(2)(conv2))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(conv3))

        # Decoder
        # Skip connections are implemented by concatenating the feature maps from the encoder path 
        # with the corresponding decoder path layers
        upconv3 = self.upconv3(bottleneck)
        conv3_1 = self.conv3_1(torch.cat([upconv3, conv3], dim=1))

        upconv2 = self.upconv2(conv3_1)
        conv2_1 = self.conv2_1(torch.cat([upconv2, conv2], dim=1))

        upconv1 = self.upconv1(conv2_1)
        conv1_1 = self.conv1_1(torch.cat([upconv1, conv1], dim=1))

        output = self.out_conv(conv1_1)
        return output


def dice_loss(pred, target, smooth=1e-6):
    # Flatten predictions and target
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def combined_loss(pred, target, weight_ce=0.5, weight_dice=0.5):
    # Cross-Entropy Loss
    ce_loss = F.cross_entropy(pred, target)
    
    # Dice Loss
    # Apply softmax to predictions and create one-hot encoding for dice loss
    pred_softmax = F.softmax(pred, dim=1)  # Assuming `pred` has shape (batch, classes, H, W)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()  # Convert to (batch, classes, H, W)
    
    dice_loss_value = dice_loss(pred_softmax, target_one_hot)

    # Combined Loss
    return weight_ce * ce_loss + weight_dice * dice_loss_value


def dice_coefficient(pred, target, num_classes, smooth=1e-6):
    """
    Calculates the Dice coefficient for multi-channel segmentation.
    Args:
        pred (torch.Tensor): Predicted tensor of shape [batch_size, num_classes, height, width]
        target (torch.Tensor): Ground truth tensor of shape [batch_size, num_classes, height, width]
        num_classes (int): Number of classes
        smooth (float): Smoothing term to avoid division by zero

    Returns:
        dice_scores (torch.Tensor): Dice scores for each class
        mean_dice (float): Mean Dice score across all classes
    """

    # Debug check for initial shapes of pred and target
    print(f"Initial pred shape: {pred.shape}")
    print(f"Initial target shape: {target.shape}")

    # Ensure `pred` and `target` are in [batch_size, num_classes, height, width] shape
    if target.dim() == 5:
        target = target.squeeze(1)  # Remove the extra dimension if present
        print(f"Target shape after squeeze: {target.shape}")

    # Make sure target has channels in the second dimension
    if target.shape[1] != num_classes:
        target = target.permute(0, 3, 1, 2)
        print(f"Target shape after permute: {target.shape}")

     # Final shape check before computing Dice
    print(f"Final pred shape: {pred.shape}")
    print(f"Final target shape: {target.shape}")
    assert pred.shape == target.shape, "Shapes of pred and target must match for Dice calculation."

    # Ensure binary masks by thresholding predictions
    pred = (pred > 0.5).float()
    
    # Initialize dice scores list
    dice_scores = []
    
    # Calculate Dice for each class
    for i in range(num_classes):
        pred_flat = pred[:, i].contiguous().view(-1)
        target_flat = target[:, i].contiguous().view(-1)

        # Debug check for flattened shapes of each channel
        print(f"Flattened pred shape for class {i}: {pred_flat.shape}")
        print(f"Flattened target shape for class {i}: {target_flat.shape}")

        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_scores.append(dice_score.item())

    # Convert to tensor for easy averaging
    dice_scores = torch.tensor(dice_scores)
    mean_dice = dice_scores.mean().item()

    return dice_scores, mean_dice