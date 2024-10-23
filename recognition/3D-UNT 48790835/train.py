import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MedicalDataset3D, load_image_paths
from modules import UNet3D
from utils import DiceLoss, calculate_dice_coefficient

# 设置超参数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
BATCH_SIZE = 1  # 根据您的显存大小调整批量大小
EPOCHS = 50

# 指定图像和标签目录
image_dir = '/Users/qiuhan/Desktop/UQ/3710/Lab3/未命名文件夹/' \
            'Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/' \
            'data/HipMRI_study_complete_release_v1/semantic_MRs_anon'

label_dir = '/Users/qiuhan/Desktop/UQ/3710/Lab3/未命名文件夹/' \
            'Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/' \
            'data/HipMRI_study_complete_release_v1/semantic_labels_anon'


# 加载图像和标签路径
image_paths, label_paths = load_image_paths(image_dir, label_dir)

# 将数据集划分为训练集和验证集
train_images, val_images, train_labels, val_labels = train_test_split(
    image_paths, label_paths, test_size=0.2, random_state=42)

# 创建数据集和数据加载器
train_dataset = MedicalDataset3D(train_images, train_labels)
val_dataset = MedicalDataset3D(val_images, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、损失函数和优化器
model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 定义训练函数
def train():
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE, dtype=torch.float)
            labels = labels.to(DEVICE, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"  Average Training Loss: {avg_train_loss:.4f}")

        # 验证步骤
        model.eval()
        val_loss = 0.0
        dice_coeffs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, dtype=torch.float)
                labels = labels.to(DEVICE, dtype=torch.float)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                dice_coeff = calculate_dice_coefficient(outputs, labels)
                dice_coeffs.append(dice_coeff)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = sum(dice_coeffs) / len(dice_coeffs)
        print(f"  Validation Loss: {avg_val_loss:.4f}, Average Dice Coefficient: {avg_dice:.4f}")


# 验证数据加载
for i in range(1):
    image, label = train_dataset[i]
    print(f"Sample {i}: Image shape: {image.shape}, Label shape: {label.shape}")

# 开始训练
if __name__ == "__main__":
    train()