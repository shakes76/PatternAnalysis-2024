from torch.utils.data import Dataset
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ADNIDataset(Dataset):
    def __init__(self, ad_path, nc_path, transform=True):
        ad_files = os.listdir(ad_path)
        nc_files = os.listdir(nc_path)
        
        
        
        ad_files_labelled = [(os.path.join(ad_path, file), 1) for file in ad_files]
        nc_files_labelled = [(os.path.join(nc_path, file), 0) for file in nc_files]
        
        self.data = ad_files_labelled + nc_files_labelled
        
        if transform:
            self.augmentations = A.Compose([
                # resize the image to 224x224
                A.LongestMaxSize(max_size=224, p=1),
                A.PadIfNeeded(min_height=224, min_width=224, value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.augmentations = A.Compose([
                A.LongestMaxSize(max_size=224, p=1),
                A.PadIfNeeded(min_height=224, min_width=224, value=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize to ImageNet standards
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file, label = self.data[item]
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = self.augmentations(image=image)['image']
        
        return image, label
        