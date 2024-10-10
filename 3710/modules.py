import nibabel as nib
import numpy as np

# 读取 NIfTI 图像
img = nib.load('/Users/zhangxiangxu/3710_data/data/HipMRI_study_complete_release_v1/semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz')

# 获取图像数据
data = img.get_fdata()

# 遍历每一层切片并打印其形状
for i in range(data.shape[2]):
    slice_shape = data[:, :, i].shape
    print(f"Shape of slice {i}: {slice_shape}")


# 获取图像数据
data = img.get_fdata()

# 打印整个图像数据的形状
print(f"Combined shape of all slices: {data.shape}")
