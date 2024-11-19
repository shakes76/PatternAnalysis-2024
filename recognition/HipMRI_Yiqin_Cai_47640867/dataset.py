import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

# Scaled image
def resize_image(image, target_shape=(256, 128)):
    resized_image = resize(image, target_shape, mode='constant', preserve_range=True)
    return resized_image

# Load 2D Nifti
def load_data_2D(image_names, norm_image=False, categorical=False, dtype=np.float32, early_stop=False):
    images = []
    affines = []
    
    num = len(image_names)
    first_case = nib.load(image_names[0]).get_fdata(caching='unchanged')
    
    # Uniform image size
    target_shape = (256, 128)  
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]

    rows, cols = target_shape
    images = np.zeros((num, rows, cols), dtype=dtype)

    for i, in_name in enumerate(tqdm(image_names)):
        nifti_image = nib.load(in_name)
        in_image = nifti_image.get_fdata(caching='unchanged')
        affine = nifti_image.affine

        if len(in_image.shape) == 3:
            in_image = in_image[:, :, 0]
        in_image = in_image.astype(dtype)

        if norm_image:
            in_image = (in_image - in_image.mean()) / in_image.std()

        if in_image.shape != target_shape:
            in_image = resize_image(in_image, target_shape)

        images[i, :, :] = in_image
        affines.append(affine)

        if i > 20 and early_stop:
            break

    return images, affines

class ProstateSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, norm_image=False, categorical=False, dtype=np.float32):
        self.image_filenames = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.label_filenames = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

        print(f"Images found in {image_dir}: {len(self.image_filenames)}")
        print(f"Labels found in {label_dir}: {len(self.label_filenames)}")

        self.images, _ = load_data_2D(self.image_filenames, norm_image=norm_image, categorical=False, dtype=dtype)
        self.labels, _ = load_data_2D(self.label_filenames, norm_image=False, categorical=categorical, dtype=dtype)

        self.labels = (self.labels > 0).astype(np.float32)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image[np.newaxis, :], label[np.newaxis, :] 
