import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import tensorflow_addons as tfa
import torchvision.transforms as transforms
import random


mrs_dir = "/Users/zhangxiangxu/3710_data/data/HipMRI_study_complete_release_v1/semantic_labels_anon"
labels_dir = "/Users/zhangxiangxu/3710_data/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"

# 自定义数据集类，用于加载 MRI 图像和对应的标签
class MRIDataset(Dataset):
    def __init__(self, mrs_dir, labels_dir, transform=None):
        self.mrs_dir = mrs_dir  # MRI 图像文件夹路径
        self.labels_dir = labels_dir  # 标签文件夹路径
        self.mr_files = sorted(os.listdir(mrs_dir))  # 获取 MRI 文件列表并排序
        self.label_files = sorted(os.listdir(labels_dir))  # 获取标签文件列表并排序
        self.transform = transform  # 数据增强变换


    def __len__(self):
        return len(self.mr_files)  # 数据集的大小，即 MRI 文件的数量


    def __getitem__(self, idx):
        # 加载 MRI 图像
        mr_path = os.path.join(self.mrs_dir, self.mr_files[idx])
        mr_image = nib.load(mr_path).get_fdata()  # 使用 nibabel 加载 NIfTI 文件并获取数据

        # 对 MRI 图像进行归一化处理，将像素值缩放到 [0, 1] 范围内
        mr_image = (mr_image - np.min(mr_image)) / (np.max(mr_image) - np.min(mr_image))

        # 加载标签图像
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label_image = nib.load(label_path).get_fdata()  # 使用 nibabel 加载标签文件并获取数据

        # 增加通道维度，使得图像形状变为 (H, W, D, C) 和 (H, W, D, 1)
        mr_image = np.expand_dims(mr_image, axis=-1)  # 增加一个通道维度，变为 (256, 256, 128, 1)
        label_image = np.expand_dims(label_image, axis=-1)
        #print(mr_image.shape, '-----------------------')
        #mr_image = np.repeat(mr_image, 128, axis=-1)
        print(mr_image.shape, '-----------------------')
        #label_image = np.repeat(label_image, 128, axis=-1)


        # 将数据转换为 PyTorch 张量
        mr_image = torch.tensor(mr_image, dtype=torch.float32)  # MRI 图像转换为 float32 类型
        label_image = torch.tensor(label_image, dtype=torch.long)  # 标签图像转换为 long 类型
        # 如果有数据增强变换，则对 MRI 和标签进行变换
        # 如果有数据增强变换，则对 MRI 进行变换
        if self.transform:
          mr_image_augmented = self.transform(mr_image)
        else:
           mr_image_augmented = mr_image

    # 返回原始 MRI 图像、增强后的 MRI 图像和标签
        return mr_image, mr_image_augmented, label_image
    


# 自定义 3D 随机翻转类，用于数据增强
class RandomFlip3D:
    def __call__(self, imgs):
        # 随机选择一个轴进行翻转
        axis = random.choice([1, 2, 3])  # 选择深度、高度或宽度轴
        imgs = torch.flip(imgs, dims=[axis])
        return imgs  # 返回翻转后的样本
    

# 自定义保持不变类，用于不做任何增强
class Identity:
    def __call__(self, imgs):
        return imgs  # 直接返回输入图像，无任何增强


# 自定义网格扭曲类，用于数据增强
class GridWarp:
    def __init__(self, grid=(4, 4, 4), max_shift=10):
        self.grid = grid
        self.max_shift = max_shift

    def griddify(self, rect, w_div, h_div):
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        x_step = w / float(w_div)
        y_step = h / float(h_div)
        y = rect[1]
        grid_vertex_matrix = []
        for _ in range(h_div + 1):
            grid_vertex_matrix.append([])
            x = rect[0]
            for _ in range(w_div + 1):
                grid_vertex_matrix[-1].append([int(x), int(y)])
                x += x_step
            y += y_step
        grid = np.array(grid_vertex_matrix)
        return grid
    def distort_grid(self, org_grid, max_shift):
        new_grid = np.copy(org_grid)
        x_min = np.min(new_grid[:, :, 0])
        y_min = np.min(new_grid[:, :, 1])
        x_max = np.max(new_grid[:, :, 0])
        y_max = np.max(new_grid[:, :, 1])
        new_grid += np.random.randint(-max_shift, max_shift + 1, new_grid.shape)
        new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
        new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
        new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
        new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
        return new_grid
    def _merge_control_points(self, grid):
      pts = grid.reshape(-1, 2)
      pts = np.array(pts, dtype='float32')[None, :, :]  # Add batch dimension
      print(f"Merged control points shape: {pts.shape}")
      return pts

    def _get_control_points(self, h, w, grid):
        x, y = grid
        dst_grid = self.griddify((0, 0, h, w), x, y)
        src_grid = self.distort_grid(dst_grid, self.max_shift)
        src = self._merge_control_points(src_grid)
        dst = self._merge_control_points(dst_grid)
        return src, dst
    
    def _get_ax_params(self, img):
        h, w, d, _ = img.shape
        params = {
            0: [
                lambda img: img, 
                self._get_control_points(w, d, (self.grid[1], self.grid[2]))
            ],
            1: [
                lambda img: tf.transpose(img, [1, 0, 2, 3]), 
                self._get_control_points(h, d, (self.grid[0], self.grid[2]))
            ],
            2: [
                lambda img: tf.transpose(img, [2, 1, 0, 3]), 
                self._get_control_points(w, h, (self.grid[1], self.grid[0]))
            ]
        }
        
        return params
    def _warp_channel(self, img, params):
        img = tf.convert_to_tensor(img, dtype='float32')
        
        def warp_axis(img, axis):
            swap, pts = params[axis]
            img = swap(img)
            src, dst = pts
            src = np.concatenate([src] * img.shape[0])
            dst = np.concatenate([dst] * img.shape[0])
            img, _ = tfa.image.sparse_image_warp(
                img, 
                src,
                dst, 
                interpolation_order=3
            )
            img = swap(img)
            return img

        for a in range(3):
            img = warp_axis(img, a)
        
        return img
    def __call__(self, imgs):
        # imgs: (H, W, D, C)
        img = imgs[0]
        print(f"Image shape before warping: {img.shape} 11111111111111111")
        h, w, d, c = imgs.shape
        params = self._get_ax_params(imgs)

        for i, img in enumerate(imgs):
            c = img.shape[-1]
            if c == 1:
                # Single channel image processing
                res = self._warp_channel(img, params)
            else:
                # Multi-channel image processing (not needed in this case)
                res = []
                for j in range(c):
                    res.append(self._warp_channel(img[:, :, j:j+1], params))
                res = tf.concat(res, axis=3)
            imgs[i] = res

        return imgs
    
# 自定义随机选择一个数据增强类
class RandomTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        transform = random.choice(self.transforms)  # 随机选择一个变换
        return transform(img)

    


# 定义新的数据增强变换组合，只随机执行其中一个变换
transform = RandomTransform([
    RandomFlip3D(),  # 使用自定义的 3D 随机翻转变换
    Identity(),      # 使用保持不变的变换
   #GridWarp(),      # 使用自定义的网格扭曲变换
])


# 创建数据集和数据加载器
dataset = MRIDataset(mrs_dir, labels_dir, transform=transform)  # 实例化数据集对象
# 创建数据加载器，用于批量加载数据，batch_size=2 表示每次加载 2 个样本
# shuffle=True 表示每次迭代时将数据集随机打乱
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

import matplotlib.pyplot as plt
# 迭代数据集，进行简单检查
for i, (mr, mr_augmented, label) in enumerate(data_loader):
    print(f"Batch {i+1} - MR shape: {mr.shape}, Augmented MR shape: {mr_augmented.shape}, Label shape: {label.shape}")
    
    # 可视化原始和增强后的图像，或打印一部分数据检查变化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    print("Original MR Slice (center):")
    axes[0, 0].imshow(mr[0, :, :, 64, 0])  # 打印第一个样本的中心切片
    print("Augmented MR Slice (center):")
    axes[0, 1].imshow(mr_augmented[0, :, :, 64, 0])  # 打印增强后的第一个样本的中心切片
    plt.tight_layout()
    plt.show()

    if i == 2:  # 仅演示前 3 批数据
        break
