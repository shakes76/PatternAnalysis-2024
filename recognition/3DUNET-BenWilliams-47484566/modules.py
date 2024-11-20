import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    #3 input and output channels
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(UNet3D, self).__init__()
        
        #Encoder definitions - downsampling
        self.encoder1 = self.conv_step(in_channels, base_filters)       
        self.encoder2 = self.conv_step(base_filters, base_filters*2)
        self.encoder3 = self.conv_step(base_filters*2, base_filters*4)
        self.encoder4 = self.conv_step(base_filters*4, base_filters*8)

        #Decoder definitons - upsampling
        self.decoder4 = self.conv_step(base_filters*16, base_filters*8)
        self.decoder3 = self.conv_step(base_filters*8, base_filters*4)
        self.decoder2 = self.conv_step(base_filters*4, base_filters*2)
        self.decoder1 = self.conv_step(base_filters*2, base_filters)
        
        
        
        #Upconvolutions
        self.upconv4 = self.upconv(base_filters*16, base_filters*8)
        self.upconv3 = self.upconv(base_filters*8, base_filters*4)
        self.upconv2 = self.upconv(base_filters*4, base_filters*2)
        self.upconv1 = self.upconv(base_filters*2, base_filters)

        #final convolution step
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

        #Bridge convolution between encoder and decoder
        self.bridge = self.conv_step(base_filters*8, base_filters*16)

    def conv_step(self, in_channels, out_channels):
        #conv + BN + relu step between feature maps
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        # Transposed convolution for upsampling
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def crop_tensor(self, input_tensor, target_tensor):
        """
        Crops the input_tensor to match the size of target_tensor along the spatial dimensions.
        Or pads if smaller
        """
        input_shape = input_tensor.shape
        target_shape = target_tensor.shape

        # Calculate the differences along each spatial dimension
        diff_depth = target_shape[2] - input_shape[2]
        diff_height = target_shape[3] - input_shape[3]
        diff_width = target_shape[4] - input_shape[4]
    
        # Pad if the input tensor is smaller than the target tensor
        pad_depth = (max(0, diff_depth // 2), max(0, diff_depth - diff_depth // 2))
        pad_height = (max(0, diff_height // 2), max(0, diff_height - diff_height // 2))
        pad_width = (max(0, diff_width // 2), max(0, diff_width - diff_width // 2))

        if diff_depth > 0 or diff_height > 0 or diff_width > 0:
            input_tensor = F.pad(input_tensor, (pad_width[0], pad_width[1], pad_height[0], pad_height[1], pad_depth[0], pad_depth[1]))

        # Crop if the input tensor is larger than the target tensor
        cropped_tensor = input_tensor[:, :,
                                  max(0, -diff_depth // 2):input_shape[2] - max(0, diff_depth - diff_depth // 2),
                                  max(0, -diff_height // 2):input_shape[3] - max(0, diff_height - diff_height // 2),
                                  max(0, -diff_width // 2):input_shape[4] - max(0, diff_width - diff_width // 2)]

        return cropped_tensor



    def forward(self, x):
        
        #Downsampling, using each encoder with a maxpool between them
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, 2))
        enc3 = self.encoder3(F.max_pool3d(enc2, 2))
        enc4 = self.encoder4(F.max_pool3d(enc3, 2))

        #Then bridge to start upsampling
        bridge = self.bridge(F.max_pool3d(enc4, 2))

        #Upsampling step, with concatenations between each upsample, downsample pair
        dec4 = self.upconv4(bridge)

        #print("dec4 size:", dec4.size())
        #print("enc4 size:", enc4.size())
        #print("Crop:", self.crop_tensor(dec4, enc4))
        dec4 = torch.cat((self.crop_tensor(dec4, enc4), enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((self.crop_tensor(dec3, enc3), enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((self.crop_tensor(dec2, enc2), enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((self.crop_tensor(dec1, enc1), enc1), dim=1)
        dec1 = self.decoder1(dec1)

        #final convolution
        out = self.final_conv(dec1)
        out = F.softmax(out, dim=1)
        return out




#This function was written by Chat GPT
def dice_coefficient_per_label(preds, targets, num_classes):
    """
    Calculate the Dice coefficient for each label.

    Args:
        preds (torch.Tensor): Predicted segmentation map (N, C, D, H, W)
        targets (torch.Tensor): Ground truth segmentation map (N, D, H, W)
        num_classes (int): Number of classes/labels

    Returns:
        dice_scores (dict): A dictionary containing Dice scores for each class
    """
    # Initialize Dice scores for each class
    dice_scores = {i: 0.0 for i in range(num_classes)}

    # Flatten the predictions and targets to be 1D
    preds = torch.argmax(preds, dim=1)  # Convert predictions to class labels
    targets = targets.view(-1)  # Flatten targets

    for label in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (preds == label).view(-1).float()
        target_mask = (targets == label).view(-1).float()

        # Calculate intersection and union
        intersection = (pred_mask * target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()

        # Calculate Dice score for the current class
        if total > 0:
            dice_scores[label] = (2. * intersection) / total

    return dice_scores




class crossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(crossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        if self.weight is not None:
            self.weight = self.weight.to(output.device)
        #target = torch.squeeze(target, dim=1)
        loss = F.cross_entropy(output, target, weight=self.weight, ignore_index=3)
        return loss
