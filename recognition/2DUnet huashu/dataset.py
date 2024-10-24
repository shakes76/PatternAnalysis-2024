import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2  # 使用OpenCV进行图像处理


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)

    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1

    return res


# 使用OpenCV进行图像缩放
def resize_image(image, target_shape=(256, 256)):
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)


def load_data_2D(imageNames, normImage=False, categorical=True, dtype=np.float32, getAffines=False, early_stop=False,
                 target_shape=(256, 256), batch_size=32):
    affines = []
    num = len(imageNames)

    if early_stop:
        num = min(20, num)  # 测试时只处理20张图像

    # 批量加载图片
    for batch_start in range(0, num, batch_size):
        batch_end = min(batch_start + batch_size, num)
        current_batch = imageNames[batch_start:batch_end]

        if categorical:
            channels = len(np.unique(nib.load(current_batch[0]).get_fdata(caching='unchanged')))
            images = np.zeros((len(current_batch), target_shape[0], target_shape[1], channels), dtype=dtype)
        else:
            images = np.zeros((len(current_batch), target_shape[0], target_shape[1]), dtype=dtype)

        for i, inName in enumerate(tqdm(current_batch)):
            niftiImage = nib.load(inName)
            inImage = niftiImage.get_fdata(caching='unchanged')
            affine = niftiImage.affine

            if len(inImage.shape) == 3:
                inImage = inImage[:, :, 0]  # 取第一层作为2D图像

            inImage = resize_image(inImage, target_shape)

            if normImage:
                inImage = (inImage - inImage.mean()) / inImage.std()

            if categorical:
                inImage = to_channels(inImage, dtype=dtype)
                images[i, :, :, :] = inImage
            else:
                images[i, :, :] = inImage

            affines.append(affine)

        yield images


def load_all_data(image_dir, normImage=False, categorical=False, dtype=np.float32, target_shape=(256, 256),
                  batch_size=32):
    imageNames = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                  f.endswith('.nii') or f.endswith('.nii.gz')]
    return load_data_2D(imageNames, normImage=normImage, categorical=categorical, dtype=dtype,
                        target_shape=target_shape, batch_size=batch_size)
