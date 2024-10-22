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
global_scores = []  #
best_images = {c: (None, -1) for c in range(6)}  # (image, score)
worst_images = {c: (None, 1) for c in range(6)}  # (image, score)
best_case = (None, -1)  # (image, score)
worst_case = (None, 1)  # (image, score)


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
            batch_scores.append(dice_score.item())

            if dice_score.item() > best_images[c][1]:
                best_images[c] = (mri_data.squeeze(0), dice_score.item())
            if dice_score.item() < worst_images[c][1]:
                worst_images[c] = (mri_data.squeeze(0), dice_score.item())
        batch_avg_dice = np.mean(batch_scores)
        global_scores.append(batch_avg_dice)
        if batch_avg_dice > best_case[1]:
            best_case = (mri_data.squeeze(0), batch_avg_dice)
        if batch_avg_dice < worst_case[1]:
            worst_case = (mri_data.squeeze(0), batch_avg_dice)


# 输出每个类的平均 Dice 分数
for c in range(6):
    print(f"Class {c} Average Dice Score: {np.mean(dice_scores[c]):.4f}")
best_avg_img, best_avg_score = best_case
worst_avg_img, worst_avg_score = worst_case

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(best_avg_img.cpu().numpy()[0], cmap='gray')
plt.title(f"Best Overall Case: Dice {best_avg_score:.4f}")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(worst_avg_img.cpu().numpy()[0], cmap='gray')
plt.title(f"Worst Overall Case: Dice {worst_avg_score:.4f}")
plt.colorbar()
plt.savefig('overall_best_worst_cases.png')
plt.show()
# 显示和保存最好和最坏的预测结果
for c in range(6):
    best_img, best_score = best_images[c]
    worst_img, worst_score = worst_images[c]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(best_img.cpu().numpy()[0], cmap='gray')
    plt.title(f"Best Case for Class {c}: Dice {best_score:.4f}")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(worst_img.cpu().numpy()[0], cmap='gray')
    plt.title(f"Worst Case for Class {c}: Dice {worst_score:.4f}")
    plt.colorbar()
    plt.savefig(f'best_worst_class_{c}.png')
    plt.show()
