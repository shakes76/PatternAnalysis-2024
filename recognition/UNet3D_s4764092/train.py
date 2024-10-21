import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt  # 用于可视化
from dataset import *
from modules import UNet3D
import numpy as np
import torch.nn.functional as F

# 设置分段分配以减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
ACCUMULATION_STEPS = 4
NUM_CLASSES = 6
Combineloss = True


# 定义设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")



class_pixel_counts = {
    0: 1068883043,
    1: 627980239,
    2: 59685345,
    3: 10172936,
    4: 2551801,
    5: 1771500,
}
total_pixels = sum(class_pixel_counts.values())

# 计算每个类别的权重
class_weights = torch.tensor(
    [total_pixels / (NUM_CLASSES * class_pixel_counts[c]) for c in range(NUM_CLASSES)],
    dtype=torch.float32,
    device=DEVICE
)

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=class_weights):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(self, inputs, targets):
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

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce = self.ce_loss(inputs, targets)
        # 计算 Dice 损失
        dice = self.dice_loss(inputs, targets)

        # 使用 log 函数应用于 Dice 损失
        log_dice = torch.log(dice + 1e-6)

        # 最终损失为 log(DiceLoss) + CeLoss
        return self.ce_weight * ce + self.dice_weight * log_dice

# 初始化模型
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)

# 设置损失函数和优化器
if CombinedLoss:
    criterion = CombinedLoss()
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
scaler = GradScaler()

# 用于记录训练和验证损失，及 Dice 系数
train_losses = []
val_losses = []
avg_dice_scores = []
epoch_dice_scores = []  # 用于保存每个 epoch 各类别的 Dice 系数
class_val_dice_scores = {c: [] for c in range(NUM_CLASSES)}
class_test_dice_scores = {c: [] for c in range(NUM_CLASSES)}
# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    optimizer.zero_grad()

    for i, (mri_data, label_data) in enumerate(train_loader):
        mri_data = torch.unsqueeze(mri_data, 1).to(DEVICE)
        label_data = label_data.to(DEVICE)

        with autocast():
            outputs = model(mri_data)
            loss = criterion(outputs, label_data)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * ACCUMULATION_STEPS

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}")

    torch.cuda.empty_cache()

    # 验证循环
    model.eval()
    val_loss = 0.0
    total_val_dice = {c: 0 for c in range(NUM_CLASSES)}
    with torch.no_grad():
        for mri_data, label_data in val_loader:
            mri_data = torch.unsqueeze(mri_data, 1).to(DEVICE)
            label_data = label_data.to(DEVICE)

            with autocast():
                outputs = model(mri_data)
                loss = criterion(outputs, label_data)
                val_loss += loss.item()

                # Compute Dice score for each class
                preds = torch.argmax(outputs, dim=1)
                for c in range(NUM_CLASSES):
                    pred_c = (preds == c).float()
                    label_c = (label_data == c).float()

                    intersection = (pred_c * label_c).sum()
                    union = pred_c.sum() + label_c.sum()
                    total_val_dice[c] += (2. * intersection + 1e-6) / (union + 1e-6)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate average Dice score for validation
        avg_dice = {c: total_val_dice[c] / len(val_loader) for c in total_val_dice}
        avg_dice_score = np.mean([avg_dice[c].item() for c in avg_dice])
        avg_dice_scores.append(avg_dice_score)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Validation Dice Score: {avg_dice_score:.4f}")
        # Output Dice score for each class
        for c in range(NUM_CLASSES):
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Class {c} Validation Dice Score: {avg_dice[c].item():.4f}")
            class_val_dice_scores[c].append(avg_dice[c].item())  # Append class-specific Dice score here
    # 每 5 个 epoch 或在最后一个 epoch 可视化一次
    if epoch == NUM_EPOCHS - 1:
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss on Validation set')
        plt.legend()
        plt.savefig(f'epoch_{epoch + 1}_1training_validation_loss.png')  # 保存图像
        plt.show()

        # 绘制 Dice 系数曲线
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(avg_dice_scores) + 1), avg_dice_scores, label='Average Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Average Validation Dice Score Over Epochs')
        plt.legend()
        plt.savefig(f'epoch_{epoch + 1}_1Validation_Dice_Score.png')  # 保存图像
        plt.show()

        plt.figure(figsize=(6, 4))
        for c in range(NUM_CLASSES):
            plt.plot(range(1, len(class_val_dice_scores[c]) + 1), class_val_dice_scores[c],
                     label=f'Class {c} Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Class Specific Validation Dice Scores Over Epochs')
        plt.legend()
        plt.savefig(f'epoch_{epoch + 1}_1class_specific_validation_dice_scores.png')  # 保存图像
        plt.show()  # 显示图像

    # 测试集评估 (用于计算 Dice 系数)
    model.eval()
    total_test_dice = {c: 0 for c in range(NUM_CLASSES)}
    with torch.no_grad():
        for mri_data, label_data in test_loader:
            mri_data = torch.unsqueeze(mri_data, 1).to(DEVICE)
            label_data = label_data.to(DEVICE)

            with autocast():
                outputs = model(mri_data)
                preds = torch.argmax(outputs, dim=1)

                for c in range(NUM_CLASSES):
                    pred_c = (preds == c).float()
                    label_c = (label_data == c).float()

                    intersection = (pred_c * label_c).sum()
                    union = pred_c.sum() + label_c.sum()
                    total_test_dice[c] += (2. * intersection + 1e-6) / (union + 1e-6)

    avg_dice = {c: total_test_dice[c] / len(test_loader) for c in total_test_dice}
    epoch_dice_scores.append(avg_dice)  # 保存当前 epoch 的各类别 Dice 系数
    avg_dice_score = np.mean([avg_dice[c].item() for c in avg_dice])
    avg_dice_scores.append(avg_dice_score)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Test Dice Score: {avg_dice_score:.4f}")

    for c in range(NUM_CLASSES):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Class {c} Test Dice Score: {avg_dice[c].item():.4f}")
        class_test_dice_scores[c].append(avg_dice[c].item())  # Append class-specific Dice score here

        # 每 5 个 epoch 或在最后一个 epoch 可视化一次
    if epoch == NUM_EPOCHS - 1:
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(avg_dice_scores) + 1), avg_dice_scores, label='Average Test Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Average Test Dice Score Over Epochs')
        plt.legend()
        plt.savefig(f'epoch_{epoch + 1}_1training_test_loss.png')  # 保存图像
        plt.show()

        plt.figure(figsize=(6, 4))
        for c in range(NUM_CLASSES):
            plt.plot(range(1, len(class_test_dice_scores[c]) + 1), class_test_dice_scores[c],
                     label=f'Class {c} Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Class Specific Dice Scores Over Epochs')
        plt.legend()
        plt.savefig(f'epoch_{epoch + 1}_1class_specific_dice_scores.png')  # 保存图像
        plt.show()  # 显示图像
