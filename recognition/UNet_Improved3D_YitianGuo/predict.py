import glob
import os
import argparse
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from dataset import MRIDataset, val_transforms  # 确保这些在 dataset.py 中正确定义
from modules import UNet3D  # 确保 UNet3D 定义在 modules.py 中
import torch.nn.functional as F
from train import split_data

# 解析命令行参数
parser = argparse.ArgumentParser(description='3D UNet Prediction Script')
parser.add_argument('--model_path', type=str, default='/home/Student/s4706162/best_model.pth')
parser.add_argument('--dataset_root', type=str, default='/home/groups/comp3710/HipMRI_Study_open')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# 使用数据划分模块
splits = split_data(args.dataset_root, seed=42, train_size=0.6, val_size=0.2, test_size=0.2)
test_image_paths, test_label_paths = splits['test']

# 初始化测试数据集
test_dataset = MRIDataset(
    image_paths=test_image_paths,
    label_paths=test_label_paths,
    transform=val_transforms,
    norm_image=True,
    dtype=np.float32
)

# 使用 DataLoader 进行数据加载
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型并加载训练好的权重
model = UNet3D().to(args.device)
model.load_state_dict(torch.load(args.model_path))
model.eval()
num_classes = model.out_channels
# 评估模型
# 初始化评估指标
dice_scores = {f'Class_{i}': [] for i in range(num_classes)}  # out_chanel 为您的类别数

with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        images = batch['image'].to(args.device, dtype=torch.float32)
        labels = batch['label'].to(args.device, dtype=torch.long)

        # 预测
        outputs = model(images)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        # 计算每个类别的 Dice 系数
        for i in range(num_classes):
            pred_i = (preds == i).cpu().numpy()
            label_i = (labels == i).cpu().numpy()
            intersection = np.logical_and(pred_i, label_i).sum()
            union = pred_i.sum() + label_i.sum()
            if union == 0:
                dice = 1.0  # 如果分母为0，说明标签和预测都为空，Dice设为1
            else:
                dice = 2 * intersection / union
            dice_scores[f'Class_{i}'].append(dice)

# 计算平均 Dice 系数
average_dice = {}
for key, value in dice_scores.items():
    average_dice[key] = np.mean(value)
    print(f"{key}: {average_dice[key]:.4f}")

# 打印总体平均 Dice 系数
overall_dice = np.mean(list(average_dice.values()))
print(f"Overall Average Dice Score: {overall_dice:.4f}")