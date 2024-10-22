import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules import UNet3D  # 确保你的模型路径正确
from dataset import *  # 你的数据集类

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = UNet3D(in_channels=1, out_channels=6)
model.load_state_dict(torch.load("model_final.pth"))
model = model.to(device)
model.eval()

# 用于存储 Dice 分数和对应的图像信息
dice_scores = {c: [] for c in range(6)}  # 假设有6个类


# 进行预测并计算 Dice 分数
with torch.no_grad():
    for mri_data, label_data in test_loader:
        mri_data = mri_data.to(device)
        label_data = label_data.to(device)
        outputs = model(mri_data)
        preds = torch.argmax(outputs, dim=1)
        batch_scores = []

        for c in range(6):
            pred_c = (preds == c).float()
            label_c = (label_data == c).float()

            intersection = (pred_c * label_c).sum()
            union = pred_c.sum() + label_c.sum()
            dice_score = (2. * intersection + 1e-6) / (union + 1e-6)

            dice_scores[c].append(dice_score.item())


# 输出每个类的平均 Dice 分数
for c in range(6):
    print(f"Class {c} Average Dice Score: {np.mean(dice_scores[c]):.4f}")


