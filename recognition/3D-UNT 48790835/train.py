import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import ImprovedUNet3D
from dataset import NiftiDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体
font_path = '/mnt/data/file-ngwyeoEN29l1M3O1QpdxCwkj'
font_prop = font_manager.FontProperties(fname=font_path)

# 定义超参数
num_epochs = 50
learning_rate = 0.001
batch_size = 1  # 由于3D数据较大，batch_size设置为1，避免内存不足

# 定义文件夹路径
labels_dir = '/Users/qiuhan/Desktop/UQ/3710/Lab3/未命名文件夹/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data/HipMRI_study_complete_release_v1/semantic_labels_anon'
mrs_dir = '/Users/qiuhan/Desktop/UQ/3710/Lab3/未命名文件夹/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon'

# 获取所有图像和标签文件
train_images = sorted([f for f in os.listdir(mrs_dir) if f.endswith('.nii')])
train_labels = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii')])

# 打印文件数量
print(f"Total images: {len(train_images)}")
print(f"Total labels: {len(train_labels)}")

train_images_filtered = []
train_labels_filtered = []

# 匹配逻辑
for image_name in train_images:
    case_id = image_name.split('_')[0] + '_' + image_name.split('_')[1]  # Case_ID

    found_match = False
    for label_name in train_labels:
        if case_id in label_name:
            train_images_filtered.append(os.path.join(mrs_dir, image_name))
            train_labels_filtered.append(os.path.join(labels_dir, label_name))
            found_match = True
            break

    if not found_match:
        print(f"未找到匹配的标签文件：{image_name}")

# 更新图像和标签文件列表
train_images = train_images_filtered
train_labels = train_labels_filtered

# 打印匹配结果
print(f"匹配的图像数量: {len(train_images)}")
print(f"匹配的标签数量: {len(train_labels)}")

# 断言检查
assert len(train_images) == len(train_labels), "图像和标签文件数量不一致！"

# 拆分训练集和验证集（90%训练，10%验证）
split_index = int(len(train_images) * 0.9)
train_images, val_images = train_images[:split_index], train_images[split_index:]
train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

# 创建数据集和数据加载器
train_dataset = NiftiDataset(train_images, train_labels)
val_dataset = NiftiDataset(val_images, val_labels)

# 打印数据集大小
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 初始化模型、损失函数和优化器
model = ImprovedUNet3D(in_channels=1, out_channels=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 如果有GPU可用，使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 训练和验证循环
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.long())
            val_running_loss += loss.item() * images.size(0)
    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_epoch_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'improved_unet3d.pth')

# 绘制损失曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch', fontproperties=font_prop)
plt.ylabel('Loss', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.savefig('loss_curve.png')
plt.close()