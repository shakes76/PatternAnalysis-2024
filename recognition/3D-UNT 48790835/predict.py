import torch
from modules import ImprovedUNet3D
from dataset import NiftiDataset
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体
font_path = '/mnt/data/file-ngwyeoEN29l1M3O1QpdxCwkj'
font_prop = font_manager.FontProperties(fname=font_path)

# 加载模型
model = ImprovedUNet3D(in_channels=1, out_channels=2)
model.load_state_dict(torch.load('improved_unet3d.pth'))
model.eval()

# 加载测试数据
test_images = ['path_to_test_image1.nii', 'path_to_test_image2.nii']  # 请根据需要替换路径
test_dataset = NiftiDataset(test_images)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 进行预测并可视化结果
with torch.no_grad():
    for idx, images in enumerate(test_loader):
        outputs = model(images)
        prediction = torch.argmax(outputs, dim=1).squeeze().numpy()

        # 可视化中间切片
        slice_idx = prediction.shape[2] // 2
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(images.squeeze().numpy()[:, :, slice_idx], cmap='gray')
        plt.title('Input Image', fontproperties=font_prop)
        plt.subplot(1, 2, 2)
        plt.imshow(prediction[:, :, slice_idx], cmap='jet')
        plt.title('Predicted Segmentation', fontproperties=font_prop)
        plt.show()

        # 保存预测结果为Nifti文件
        pred_img = nib.Nifti1Image(prediction.astype(np.uint8), affine=np.eye(4))
        nib.save(pred_img, f'prediction_{idx}.nii')