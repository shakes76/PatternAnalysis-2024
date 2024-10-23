import torch
from torch.utils.data import DataLoader
from modules import UNet3D
from dataset import MedicalDataset3D, load_image_paths
import matplotlib.pyplot as plt
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载测试数据集
# predict.py

from dataset import MedicalDataset3D, load_image_paths

# 设置数据路径
data_dir = '/Users/qiuhan/Desktop/UQ/3710/Lab3/未命名文件夹/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data'


# 获取测试集的文件路径
test_image_paths, test_label_paths = load_image_paths(data_dir, split='test')

# 创建测试数据集
test_dataset = MedicalDataset3D(
    image_paths=test_image_paths,
    label_paths=test_label_paths,
    normImage=True,
    categorical=False,
    dtype=np.float32,
    orient=True  # 如果需要应用方向和缩放
)

# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型并加载训练好的权重
model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load('checkpoint_epoch_50.pth', map_location=DEVICE))
model.eval()

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()

dice_scores = []

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)

        predictions = model(data)
        dice = dice_coefficient(torch.sigmoid(predictions), targets)
        dice_scores.append(dice)

        # 可视化结果（仅展示中间的一张切片）
        slice_idx = data.shape[2] // 2
        input_slice = data.cpu().numpy()[0, 0, slice_idx, :, :]
        target_slice = targets.cpu().numpy()[0, 0, slice_idx, :, :]
        pred_slice = torch.sigmoid(predictions).cpu().numpy()[0, 0, slice_idx, :, :]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(input_slice, cmap='gray')
        axes[0].set_title('Input')
        axes[1].imshow(target_slice, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[2].imshow(pred_slice, cmap='gray')
        axes[2].set_title('Prediction')
        plt.show()

# 打印平均 Dice 系数
avg_dice = np.mean(dice_scores)
print(f'Average Dice Similarity Coefficient on Test Set: {avg_dice:.4f}')

