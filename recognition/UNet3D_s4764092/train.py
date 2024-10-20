import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt  # 用于可视化
from dataset import *
from modules import UNet3D
import numpy as np

# 设置分段分配以减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
ACCUMULATION_STEPS = 4
NUM_CLASSES = 6


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

# 初始化模型
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)

# 设置损失函数和优化器
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


