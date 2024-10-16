'''preprocess dataset'''
import pandas as pd
import torch
from torch.utils.data import DataLoader, dataloader, Dataset
import os
from torchvision import transforms, io
from PIL import Image
#import matplotlib as plt
import matplotlib.pyplot as plt
import random


SCALE_FACTOR = 1.0/255

current_dir = os.getcwd()
excel = os.path.join(current_dir,'recognition','45813788_Siamese','dataset', 'train-metadata.csv')
images = os.path.join(current_dir,'recognition','45813788_Siamese','dataset', 'train-image','image')

df = pd.read_csv(excel)

df= df.drop(columns=['Unnamed: 0'])
df= df.drop(columns=['patient_id'])

benign = df[df['target'] == 0].reset_index(drop=True)
malignant = df[df['target'] == 1].reset_index(drop=True)

#augument dataset since theres only 584 malignant and 30000 non malig
malig_aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    #normalise using ImageNet (pytorch defaults)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    #could add randomRotation but leave it like this for now
    #could add color jitter too but i think this will do more harm then good
])

#augument dataset since theres only 584 malignant and 30000 non malig
benign_aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    #normalise using ImageNet (pytorch defaults)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Get the path to the image
image_name = benign['isic_id'][0]  # Replace with malignant['isic_id'][0] if you want a malignant image
image_path = os.path.join(images, f"{image_name}.jpg")

# Open the image using PIL
image = Image.open(image_path)

# Apply the transformation (use benign_aug or malig_aug depending on the target)
augmented_image = benign_aug(image)  # For benign; use malig_aug for malignant

# Convert the tensor back to a PIL image for display (undo normalization for display purposes)
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = std * tensor + mean  # unnormalize
    tensor = tensor.clip(0, 1)  # clip values to be in range [0, 1]
    return tensor

# Convert the augmented tensor to an image
augmented_image_np = tensor_to_image(augmented_image)

# Display the image
plt.imshow(augmented_image_np)
plt.axis('off')  # Turn off axis labels
plt.show()


class ISISCDataset(Dataset):
    def __init__(self, benign_df, malignant_df, images_dir, transform_benign=None, transform_malignant=None, augment_ratio=1.0):
        self.benign_df = benign_df
        self.malignant_df = malignant_df
        self.images_dir = images_dir
        self.transform_benign = transform_benign
        self.transform_malignant = transform_malignant
        self.augment_ratio = augment_ratio

        #list of image IDs
        self.benign_ids = self.benign_df['isic_id'].tolist()
        self.malignant_ids = self.malignant_df['isic_id'].tolist()

        #Gen paris
        self.pairs, self.labels = self.create_pairs()

    def create_pairs(self):
        pairs = []
        labels = []

        #use 1 for similar and 0 for not 

        for _ in range(len(self.benign_ids)):
            img1, img2 = random.sample(self.benign_ids, 2)
            pairs.append((img1, img2))
            labels.append(1)

        for _ in range(len(self.malignant_ids)):
            img1, img2 = random.sample(self.malignant_ids, 2)
            pairs.append((img1, img2))
            labels.append(1)

            if self.augment_ratio > 0 and random.random() < self.augment_ratio:
                pairs.append((img1,img2))
                labels.append(1)

        # Negative Pairs
        for _ in range(len(self.benign_ids)):
            img1 = random.choice(self.benign_ids)
            img2 = random.choice(self.benign_ids)
            pairs.append((img1, img2))
            labels.append(0)

        return pairs, labels

    def __len__(self):
        return len(self.pairs)

    def apply_transform(self, img_id, img):
        # we only need to look at malignant class and if its not in there
        # then we carry on
        if img_id in self.malignant_ids:
            img = self.transform_malignant(img)
        else:
            if self.transform_benign:
                img = self.transform_benign(img)

        return img

    def __getitem__(self, index):
        img1_id, img2_id = self.pairs[index]
        label = self.labels[index]

        #get images
        img1_path = os.path.join(self.images_dir,img1_id + '.jpg')
        img2_path = os.path.join(self.images_dir,img1_id + '.jpg')

        #normalise pixel values
        img1 = io.read_image(img1_path).float() * SCALE_FACTOR
        img2 = io.read_image(img2_path).float() * SCALE_FACTOR

        #transform the images
        img1 = self.apply_transform(img1_id, img1)
        img2 = self.apply_transform(img2_id, img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

#print(malignant)
#print(benign)

# Check if the path exists