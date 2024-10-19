import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from utils import IMAGE_HEIGHT, IMAGE_WIDTH
from utils import TRAIN_IMG_DIR, TRAIN_MASK_DIR

def load_data_2D(imageNames, normImage=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0 to 1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    """
    num = len(imageNames)
    images = np.zeros((num, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=dtype)
    affines = []

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # Read disk only

        affine = niftiImage.affine
        affines.append(affine)

        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # Sometimes extra dims in HipMRI study data -> remove
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        inImage_height, inImage_width = inImage.shape # Note: some images smaller than 256x144
        images[i,:inImage_height,:inImage_width] = inImage
        
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def get_names(data_path):
    names = [os.path.join(data_path, img) for img in os.listdir(data_path) if img.endswith(('.nii', '.nii.gz'))]
    return names

class ProstateDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms, early_stop=False):
        self.imageNames = get_names(image_dir)
        self.maskNames = get_names(mask_dir)
        self.images = load_data_2D(self.imageNames, normImage=True, early_stop=early_stop) # Input images normalised
        self.masks = load_data_2D(self.maskNames, early_stop=early_stop)
        self.transforms = transforms

    def __len__(self):
		# Total number of samples contained in the dataset
        return len(self.imageNames)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
		
        if self.transforms is not None:
			# Apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
		
        return (image, mask)

def test():
    imageNames = get_names(TRAIN_IMG_DIR)
    images = load_data_2D(imageNames, normImage=False)
    print(images.shape)
    plt.imshow(images[11000])
    plt.show()

    maskNames = get_names(TRAIN_MASK_DIR)
    masks = load_data_2D(maskNames, normImage=False)
    print(masks.shape)
    plt.imshow(masks[11000])
    plt.show()

if __name__ == "__main__":
    test()