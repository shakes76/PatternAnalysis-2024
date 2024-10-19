import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from tqdm import tqdm

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32):
    images = []
    for inName in tqdm(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] 
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        images.append(inImage)
    return np.array(images)

class HipMRIDataset(Dataset):
    def __init__(self, data_dir, seg_dir, transform=None):
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        seg_path = os.path.join(self.seg_dir, self.image_files[idx])
        
        image = load_data_2D([img_path], normImage=True)[0]
        mask = load_data_2D([seg_path], dtype=np.uint8)[0]

        # Ensure image and mask are 2D
        if len(image.shape) > 2:
            image = image[:,:,0]
        if len(mask.shape) > 2:
            mask = mask[:,:,0]

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask