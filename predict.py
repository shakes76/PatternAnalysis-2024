import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import UNet2D
from dataset import load_all_data

# 计算Dice系数的函数
def calculate_dice_score(pred, target, smooth=1e-6):
    pred = pred > 0.5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 预测函数
def predict(model, test_data):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for i in range(test_data.shape[0]):
            inputs = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            outputs = model(inputs)
            outputs = (outputs > 0.5).float()

            # 使用真实标签（输入数据）计算 Dice 系数
            dice_score = calculate_dice_score(outputs, inputs)
            dice_scores.append(dice_score.item())

            # 可视化输入图像和预测结果
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            plt.imshow(inputs.squeeze().numpy(), cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("Predicted Mask")
            plt.imshow(outputs[0, 0].cpu().numpy(), cmap='gray')  # 显示第一个通道
            plt.show()

    avg_dice_score = np.mean(dice_scores)
    print(f'Average Dice Coefficient on test set: {avg_dice_score:.4f}')

if __name__ == '__main__':
    model = UNet2D(in_channels=1, out_channels=2)  # 使用与训练时相同的通道数
    model.load_state_dict(torch.load('unet_model1.pth'))

    image_dir = r'D:\桌面\HipMRI_study_keras_slices_data\keras_slices_test'  # 修正路径
    test_data = next(load_all_data(image_dir, normImage=True, target_shape=(256, 256)))

    predict(model, test_data)
