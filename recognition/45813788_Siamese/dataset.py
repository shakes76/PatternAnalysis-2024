'''preprocess dataset'''
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

#default transform
default_aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


#augument dataset since theres only 584 malignant and 30000 non malig
train_aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])


class ISICDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, augment_ratio=1.0):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.augment_ratio = augment_ratio

        #list of image IDs
        self.image_ISIC = df['isic_id'].tolist()
        self.labels = df['target'].astype(int).tolist()

    def __len__(self):
        return len(self.image_ISIC)

    def __getitem__(self, index):
        img_id = self.image_ISIC[index]    
        label = self.labels[index]
       
        #get images
        img_path = os.path.join(self.images_dir,img_id + '.jpg')
        
        #normalise pixel values
        img = Image.open(img_path).convert('RGB')


        #transform the images
        if self.transform:
            img = self.transform(img)
        else:
            img = default_aug(img)
     
        
        return img, torch.tensor(label, dtype=torch.long)

def load_data(excel):
    df = pd.read_csv(excel)
    df = df.drop(columns=['Unnamed: 0', 'patient_id'])
    return df

def split_data(df, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, random_state=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=(1-train_ratio),
        stratify=df['target'],
        random_state=random_state
    )

    #split temp into training and validation
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df['target'],
        random_state=random_state
    )

    return train_df, val_df, test_df