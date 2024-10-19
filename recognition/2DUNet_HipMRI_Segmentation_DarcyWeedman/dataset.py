import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from tqdm import tqdm

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32):
    images = []
    for inName in tqdm(imageNames):
        try:
            niftiImage = nib.load(inName)
            inImage = niftiImage.get_fdata(caching='unchanged')
            if len(inImage.shape) == 3:
                inImage = inImage[:,:,0]  # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
            if normImage:
                inImage = (inImage - inImage.mean()) / inImage.std()
            images.append(inImage)
        except FileNotFoundError:
            print(f"File not found: {inName}")
    return np.array(images)

class HipMRIDataset(Dataset):
    def __init__(self, data_dir, seg_dir=None, transform=None):
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = load_data_2D([img_path], normImage=True)[0]

        # Add channel dimension
        image = np.expand_dims(image, axis=0)

        if self.seg_dir:
            seg_filename = self.image_files[idx].replace('case_', 'seg_')
            seg_path = os.path.join(self.seg_dir, seg_filename)
            if os.path.exists(seg_path):
                mask = load_data_2D([seg_path], dtype=np.uint8)[0]
                # Ensure mask is 2D
                if len(mask.shape) > 2:
                    mask = mask[:,:,0]
                mask = np.expand_dims(mask, axis=0)
            else:
                print(f"Segmentation file not found: {seg_path}")
                mask = np.zeros_like(image)
        else:
            mask = np.zeros_like(image)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_data_dir = os.path.join(base_dir, "keras_slices_train")
    train_seg_dir = os.path.join(base_dir, "keras_slices_seg_train")
    val_data_dir = os.path.join(base_dir, "keras_slices_validate")
    val_seg_dir = os.path.join(base_dir, "keras_slices_seg_validate")
    test_data_dir = os.path.join(base_dir, "keras_slices_test")
    test_seg_dir = os.path.join(base_dir, "keras_slices_seg_test")

    train_dataset = HipMRIDataset(train_data_dir, train_seg_dir)
    val_dataset = HipMRIDataset(val_data_dir, val_seg_dir)
    test_dataset = HipMRIDataset(test_data_dir, test_seg_dir)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("\nChecking data directories:")
    for dir_name in [train_data_dir, train_seg_dir, val_data_dir, val_seg_dir, test_data_dir, test_seg_dir]:
        print(f"{dir_name}: {'Exists' if os.path.exists(dir_name) else 'Does not exist'}")

    sample_image, sample_mask = train_dataset[0]
    print(f"\nSample image shape: {sample_image.shape}")
    print(f"Sample mask shape: {sample_mask.shape}")

    print("\nFirst few files in train data directory:")
    for file in sorted(os.listdir(train_data_dir))[:5]:
        print(file)

    if os.path.exists(train_seg_dir):
        print("\nFirst few files in train segmentation directory:")
        for file in sorted(os.listdir(train_seg_dir))[:5]:
            print(file)
    else:
        print(f"\nTrain segmentation directory does not exist: {train_seg_dir}")