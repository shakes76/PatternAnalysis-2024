import torch
import torch.nn as nn
import torch.nn.functional as F

class_pixel_counts = {
    0: 1068883043,
    1: 627980239,
    2: 59685345,
    3: 10172936,
    4: 2551801,
    5: 1771500,
}
total_pixels = sum(class_pixel_counts.values())

# 定义设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")

NUM_CLASSES=6
# 计算每个类别的权重
class_weights = torch.tensor(
    [total_pixels / (NUM_CLASSES * class_pixel_counts[c]) for c in range(NUM_CLASSES)],
    dtype=torch.float32,
    device=DEVICE
)

def weighted_cross_entropy_loss():
    """
    返回一个加权交叉熵损失函数
    :param class_weights: 类别的权重
    :param device: GPU 或 CPU
    :return: 带有类别权重的交叉熵损失
    """
    return nn.CrossEntropyLoss(weight=class_weights)

class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        smooth = 1e-6
        # 计算 softmax 后的概率
        inputs_softmax = torch.softmax(inputs, dim=1)
        # 将 targets 转换为 one-hot 编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        # 计算交集和并集
        intersection = torch.sum(inputs_softmax * targets_one_hot, dim=(2, 3, 4))
        union = torch.sum(inputs_softmax, dim=(2, 3, 4)) + torch.sum(targets_one_hot, dim=(2, 3, 4))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        print(dice)
        return 1 - torch.mean(dice)

import torch
import torch.nn as nn
import torch.nn.functional as F

class_pixel_counts = {
    0: 1068883043,
    1: 627980239,
    2: 59685345,
    3: 10172936,
    4: 2551801,
    5: 1771500,
}
total_pixels = sum(class_pixel_counts.values())

# 定义设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")

NUM_CLASSES = 6
# 计算每个类别的权重
class_weights = torch.tensor(
    [total_pixels / (NUM_CLASSES * class_pixel_counts[c]) for c in range(NUM_CLASSES)],
    dtype=torch.float32,
    device=DEVICE
)

def weighted_cross_entropy_loss():
    """
    返回一个加权交叉熵损失函数
    :return: 带有类别权重的交叉熵损失实例
    """
    return nn.CrossEntropyLoss(weight=class_weights)

class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        smooth = 1e-6
        # 计算 softmax 后的概率
        inputs_softmax = torch.softmax(inputs, dim=1)
        # 将 targets 转换为 one-hot 编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        # 计算交集和并集
        intersection = torch.sum(inputs_softmax * targets_one_hot, dim=(2, 3, 4))
        union = torch.sum(inputs_softmax, dim=(2, 3, 4)) + torch.sum(targets_one_hot, dim=(2, 3, 4))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - torch.mean(dice)


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = weighted_cross_entropy_loss()  # 实例化交叉熵损失
        self.dice_loss = DiceLoss()  # 实例化 Dice 损失

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce = self.ce_loss(inputs, targets)
        # 计算 Dice 损失
        dice = self.dice_loss(inputs, targets)

        # 使用 log 函数应用于 Dice 损失
        log_dice = torch.log(dice + 1e-6)

        # 最终损失为 log(DiceLoss) + CeLoss
        return self.ce_weight * ce + self.dice_weight * log_dice
