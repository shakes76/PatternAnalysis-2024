import os
import glob
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRIDataset, train_transforms, val_transforms  # 确保这些在 dataset.py 中正确定义
from modules import UNet3D  # 确保 UNet3D 定义在 modules.py 中
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import nibabel as nib

# 设置随机种子以保证结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 解析命令行参数
parser = argparse.ArgumentParser(description='3D UNet Training Script')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--loss', type=str, default='dice', choices=['mse', 'dice', 'ce', 'combined'])
parser.add_argument('--dataset_root', type=str, default='/home/groups/comp3710/HipMRI_Study_open')
args = parser.parse_args()
# 获取图像和标签路径
image_paths = sorted(glob.glob(os.path.join(args.dataset_root, 'semantic_MRs', '*.nii.gz')))
label_paths = sorted(glob.glob(os.path.join(args.dataset_root, 'semantic_labels_only', '*.nii.gz')))

# 确保图像和标签数量一致
assert len(image_paths) == len(label_paths), "图像和标签数量不一致"

# 将数据集划分为训练集和验证集
train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(
    image_paths, label_paths, test_size=0.3, random_state=42
)

# 初始化训练和验证数据集
train_dataset = MRIDataset(
    image_paths=train_image_paths,
    label_paths=train_label_paths,
    transform=train_transforms,
    norm_image=True,
    dtype=np.float32
)

val_dataset = MRIDataset(
    image_paths=val_image_paths,
    label_paths=val_label_paths,
    transform=val_transforms,
    norm_image=True,
    dtype=np.float32
)

# 使用 DataLoader 进行数据加载
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

def compute_class_weights(label_paths, num_classes):
    class_counts = np.zeros(num_classes)
    for path in label_paths:
        label = nib.load(path).get_fdata().astype(np.uint8)
        unique, counts = np.unique(label, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[u] += c
    class_weights = class_counts.max() / class_counts
    return torch.tensor(class_weights, dtype=torch.float32).to(args.device)

# 定义DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        # 使用Softmax获取每个类别的概率
        inputs = F.softmax(inputs, dim=1)
        # 将targets转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        # 计算每个类别的Dice损失
        intersection = (inputs * targets_one_hot).sum(dim=(2,3,4))
        union = inputs.sum(dim=(2,3,4)) + targets_one_hot.sum(dim=(2,3,4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        # 排除背景类别（索引为0）
        dice = dice[:, 1:]
        # 计算平均Dice损失
        dice_loss = 1 - dice.mean()
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        # 直接使用 targets，不需要 argmax
        loss_ce = self.ce(inputs, targets.squeeze(1))
        loss_dice = self.dice(inputs, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice

# 初始化模型并移动到设备
model = UNet3D().to(args.device)

# 计算类别权重
num_classes = model.final_conv.out_channels
class_weights = compute_class_weights(train_label_paths, num_classes)

# 定义损失函数
if args.loss == 'dice':
    criterion = DiceLoss().to(args.device)
elif args.loss == 'ce':
    criterion = nn.CrossEntropyLoss().to(args.device)
elif args.loss == 'combined':
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=1.0).to(args.device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 初始化GradScaler用于混合精度训练
scaler = torch.cuda.amp.GradScaler()

# 初始化损失和指标记录列表
train_losses = []
valid_losses = []
train_dice_scores = []
valid_dice_scores = []
# 新增的初始化列表
val_losses = []
val_dices = []

def plot_metrics(train_losses, val_losses, train_dices, val_dices, save_path='metrics.png'):
    """绘制训练和验证损失以及Dice系数"""
    epochs = range(1, len(train_losses) + 1)  # 现在长度等于 epoch 的数量

    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()

    # 绘制Dice系数
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dices, 'b-', label='Training Dice')
    plt.plot(epochs, val_dices, 'r-', label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient During Training')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    num_classes = model.final_conv.out_channels  # 获取类别数量
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)  # [B, 1, D, H, W]
            labels = batch['label'].to(device).long()  # [B, D, H, W] 或 [B, 1, D, H, W]

            if labels.dim() == 5 and labels.size(1) == 1:
                labels = labels.squeeze(1)  # 现在 labels 是 [B, D, H, W]

            with autocast():
                outputs = model(images)  # [B, num_classes, D, H, W]
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            # 计算预测结果
            preds = torch.argmax(outputs, dim=1)  # [B, D, H, W]

            # 现在 preds 和 labels 都是 [B, D, H, W]
            targets = labels

            # 计算每个类别的 Dice 系数
            dice_scores = []
            for class_index in range(num_classes):
                pred_class = (preds == class_index).float()
                target_class = (targets == class_index).float()

                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum()
                if union == 0:
                    dice = 1.0  # 如果分母为0，表示没有该类别，设定 dice 为1
                else:
                    dice = (2. * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())

            # 计算平均 Dice 系数（如果需要，可以排除背景类）
            avg_dice_score = np.mean(dice_scores[1:])  # 排除背景类（索引为0）
            val_dice += avg_dice_score

    avg_loss = val_loss / len(val_loader)
    avg_dice = val_dice / len(val_loader)
    return avg_loss, avg_dice

def main():
    start_time = time.time()
    num_epochs = args.epoch
    best_val_dice = 0
    save_path = 'best_model.pth'
    num_classes = model.final_conv.out_channels

    print("begin training")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = []
        epoch_train_dice = []

        for idx, batch in enumerate(train_dataloader):
            images = batch['image'].to(args.device, dtype=torch.float32)
            labels = batch['label'].to(args.device, dtype=torch.long)

            if labels.dim() == 5 and labels.size(1) == 1:
                labels = labels.squeeze(1)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss.append(loss.item())

            # 计算Dice系数
            preds = torch.argmax(outputs, dim=1)
            targets = labels

            dice_scores = []
            for class_index in range(num_classes):
                pred_class = (preds == class_index).float()
                target_class = (targets == class_index).float()

                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum()
                if union == 0:
                    dice = 1.0
                else:
                    dice = (2. * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())

            avg_dice_score = np.mean(dice_scores[1:])  # 排除背景类
            epoch_train_dice.append(avg_dice_score)

            # 打印训练进度
            if (idx + 1) % max(1, len(train_dataloader) // 2) == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{idx + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, Elapsed Time: {elapsed:.2f}s"
                )

        # 在每个 epoch 结束时记录平均损失和 Dice 系数
        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_dice = np.mean(epoch_train_dice)
        train_losses.append(avg_train_loss)
        train_dice_scores.append(avg_train_dice)

        # 验证阶段
        avg_val_loss, avg_val_dice = evaluate(model, val_loader, criterion, args.device)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        print(
            f"Epoch [{epoch}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, "
            f"Training Dice: {avg_train_dice:.4f}, Validation Loss: {avg_val_loss:.4f}, "
            f"Validation Dice: {avg_val_dice:.4f}"
        )

        # 保存最好的模型
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
            print(f"Saved the best model, Validation Dice: {best_val_dice:.4f}")

    # 绘制损失和Dice系数
    plot_metrics(train_losses, val_losses, train_dice_scores, val_dices, save_path='training_metrics.png')

    print("Finished Training")

if __name__ == '__main__':
    main()
