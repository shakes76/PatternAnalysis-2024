import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from modules import UNet3D  # 确保你的模型路径正确
from dataset import *  # 你的数据集类
import os
from torch.cuda.amp import GradScaler, autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = UNet3D(in_channels=1, out_channels=6)
model.load_state_dict(torch.load("/home/Student/s4764092/final.pth"))
model = model.to(DEVICE)
model.eval()

# 用于存储 Dice 分数和对应的图像信息
NUM_CLASSES = 6
dice_scores = {c: [] for c in range(NUM_CLASSES)}  # 每个类的 Dice 分数列表
best_images = {c: (None, -1) for c in range(NUM_CLASSES)}  # (image, best_score)
worst_images = {c: (None, 1) for c in range(NUM_CLASSES)}  # (image, worst_score)

total_test_dice = {c: 0 for c in range(NUM_CLASSES)}
avg_dice_scores = []

# 全局最好的和最坏的案例（综合Dice）
best_case = (None, -1)  # (image, score)
worst_case = (None, 1)  # (image, score)
global_scores = []
# 进行预测并计算 Dice 分数
with torch.no_grad():
    for mri_data, label_data in test_loader:
        mri_data = torch.unsqueeze(mri_data, 1).to(DEVICE)
        label_data = label_data.to(DEVICE)

        with autocast():
            outputs = model(mri_data)
            preds = torch.argmax(outputs, dim=1)

            # 计算每个类的 Dice 分数
            batch_dice_scores = []
            for c in range(NUM_CLASSES):
                pred_c = (preds == c).float()
                label_c = (label_data == c).float()

                intersection = (pred_c * label_c).sum()
                union = pred_c.sum() + label_c.sum()
                dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
                total_test_dice[c] += dice_score

                # 记录最佳和最差的预测
                if dice_score > best_images[c][1]:
                    best_images[c] = (mri_data.cpu().numpy(), dice_score.item())
                if dice_score < worst_images[c][1]:
                    worst_images[c] = (mri_data.cpu().numpy(), dice_score.item())

                # 存储该类别的 Dice 分数用于全局比较
                batch_dice_scores.append(dice_score.item())

            # 计算批次的平均 Dice 分数
            batch_avg_dice = np.mean(batch_dice_scores)

            # 更新全局最好的和最差的案例
            if batch_avg_dice > best_case[1]:
                best_case = (mri_data.cpu().numpy(), batch_avg_dice)
            if batch_avg_dice < worst_case[1]:
                worst_case = (mri_data.cpu().numpy(), batch_avg_dice)

# 计算每个类的平均 Dice 分数
avg_dice = {c: total_test_dice[c] / len(test_loader) for c in total_test_dice}
avg_dice_score = np.mean([avg_dice[c].item() for c in avg_dice])
avg_dice_scores.append(avg_dice_score)

# 输出每个类的平均 Dice 分数
for c in range(NUM_CLASSES):
    print(f"Class {c}: Average Dice Score = {avg_dice[c].item():.4f}")

# 输出总的平均 Dice 分数
print(f"Overall Average Dice Score: {avg_dice_score:.4f}")



# 定义类别的颜色映射，使用不同颜色代表不同类别
colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple']  # 为6个类别定义颜色
cmap = ListedColormap(colors[:NUM_CLASSES])

# 定义可视化函数
def visualize_image_and_prediction(img, pred, title_img, title_pred,save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原始MRI图像（最大投影）
    img_proj = np.max(img[0, 0, :, :, :], axis=-1)  # 投影
    axs[0].imshow(img_proj, cmap='gray')
    axs[0].set_title(title_img)

    # 显示预测图像（最大投影并且用不同颜色表示每个类别）
    pred_proj = np.max(pred[0, :, :, :], axis=-1)  # 投影预测图
    axs[1].imshow(pred_proj, cmap=cmap, interpolation='nearest')
    axs[1].set_title(title_pred)
    plt.savefig(save_path)
    plt.close(fig)  # 关闭图像，防止显示



# 输出每个类的最佳和最差的预测图像
for c in range(NUM_CLASSES):
    best_img, best_score = best_images[c]
    worst_img, worst_score = worst_images[c]

    if best_img is not None and worst_img is not None:
        # 获取最佳和最差的预测结果
        best_pred = torch.argmax(model(torch.from_numpy(best_img).to(DEVICE)), dim=1).cpu().numpy()
        worst_pred = torch.argmax(model(torch.from_numpy(worst_img).to(DEVICE)), dim=1).cpu().numpy()

        # 可视化最佳预测
        visualize_image_and_prediction(
            best_img, best_pred,
            title_img=f'Best Original Image (Class {c}): Dice = {best_score:.4f}',
            title_pred=f'Best Prediction (Class {c}): Dice = {best_score:.4f}',
            save_path = f'best_class_{c}.png'
        )

        # 可视化最差预测
        visualize_image_and_prediction(
            worst_img, worst_pred,
            title_img=f'Worst Original Image (Class {c}): Dice = {worst_score:.4f}',
            title_pred=f'Worst Prediction (Class {c}): Dice = {worst_score:.4f}',
            save_path=f'worst_class_{c}.png'
        )

# 可视化并输出总的最好的和最差的案例（全局平均 Dice）
if best_case[0] is not None and worst_case[0] is not None:
    # 获取最佳和最差的全局预测结果
    best_pred_global = torch.argmax(model(torch.from_numpy(best_case[0]).to(DEVICE)), dim=1).cpu().numpy()
    worst_pred_global = torch.argmax(model(torch.from_numpy(worst_case[0]).to(DEVICE)), dim=1).cpu().numpy()

    # 可视化最佳全局案例
    visualize_image_and_prediction(
        best_case[0], best_pred_global,
        title_img=f'Best Case: Overall Average Dice = {best_case[1]:.4f}',
        title_pred=f'Best Prediction: Overall Average Dice = {best_case[1]:.4f}',
        save_path='best_global.png'
    )

    # 可视化最差全局案例
    visualize_image_and_prediction(
        worst_case[0], worst_pred_global,
        title_img=f'Worst Case: Overall Average Dice = {worst_case[1]:.4f}',
        title_pred=f'Worst Prediction: Overall Average Dice = {worst_case[1]:.4f}',
        save_path = 'worst_global.png'
    )
