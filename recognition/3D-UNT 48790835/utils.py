import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        num_classes = outputs.shape[1]
        outputs = F.softmax(outputs, dim=1)

        # 将targets转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]

        outputs = outputs.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)

        intersection = (outputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets_one_hot.sum() + self.smooth)
        loss = 1 - dice

        return loss

def calculate_dice_coefficient(outputs, targets):
    num_classes = outputs.shape[1]
    outputs = F.softmax(outputs, dim=1)
    preds = torch.argmax(outputs, dim=1)

    dice_score = 0.0
    for i in range(num_classes):
        pred_i = (preds == i).float()
        target_i = (targets == i).float()
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        dice = (2. * intersection) / (union + 1e-5)
        dice_score += dice

    return dice_score / num_classes