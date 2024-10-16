import glob
import os
import argparse
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MRIDataset, val_transforms  # 确保这些在 dataset.py 中正确定义
from modules import UNet3D  # 确保 UNet3D 定义在 modules.py 中

# 解析命令行参数
parser = argparse.ArgumentParser(description='3D UNet Prediction Script')
parser.add_argument('--model_path', type=str, default='/home/Student/s4706162/best_model.pth')
parser.add_argument('--dataset_root', type=str, default='/home/groups/comp3710/HipMRI_Study_open')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# 获取图像和标签路径
image_paths = sorted(glob.glob(os.path.join(args.dataset_root, 'semantic_MRs', '*.nii.gz')))
label_paths = sorted(glob.glob(os.path.join(args.dataset_root, 'semantic_labels_only', '*.nii.gz')))

# 初始化验证数据集
val_dataset = MRIDataset(
    image_paths=image_paths,
    label_paths=label_paths,
    transform=val_transforms,
    norm_image=True,
    dtype=np.float32
)

# 使用 DataLoader 进行数据加载
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 初始化模型并加载训练好的权重
model = UNet3D().to(args.device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# 评估模型
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        images = batch['image'].to(args.device, dtype=torch.float32)
        labels = batch['label'].to(args.device, dtype=torch.long)

        # 预测
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        # 可视化结果
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(images[0, 0, :, :, images.shape[4] // 2], cmap='gray')
        ax[0].set_title('Input Image')
        ax[1].imshow(labels[0, :, :, labels.shape[3] // 2], cmap='gray')
        ax[1].set_title('Ground Truth')
        ax[2].imshow(preds[0, :, :, preds.shape[3] // 2], cmap='gray')
        ax[2].set_title('Prediction')
        plt.show()

        # 仅展示一个样本
        break