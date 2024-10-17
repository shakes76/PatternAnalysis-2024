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
        self.pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []
        labels1 = []
        labels2 = []

        #0 for malig, 1 for benign
        for _ in range(len(self.benign_ids)):
            img1, img2 = random.sample(self.benign_ids, 2)
            pairs.append((img1, img2))
            labels1.append(0)
            labels2.append(0)

        for _ in range(len(self.malignant_ids)):
            img1, img2 = random.sample(self.malignant_ids, 2)
            pairs.append((img1, img2))
            labels1.append(1)
            labels2.append(1)

        # Negative Pairs
        for _ in range(len(self.benign_ids)):
            img1 = random.choice(self.benign_ids)
            img2 = random.choice(self.malignant_ids) #change if bad
            pairs.append((img1, img2))
            labels1.append(0)
            labels2.append(1)

        self.labels1 = labels1
        self.labels2 = labels2

        return pairs

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
        
        label1 = self.labels1[index]
        label2 = self.labels2[index]

        #get images
        img1_path = os.path.join(self.images_dir,img1_id + '.jpg')
        img2_path = os.path.join(self.images_dir,img2_id + '.jpg')

        #normalise pixel values
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')


        #transform the images
        img1 = self.apply_transform(img1_id, img1)
        img2 = self.apply_transform(img2_id, img2)
        
        return img1, img2, torch.tensor(label1, dtype=torch.float32), torch.tensor(label2, dtype=torch.float32)

#print(malignant)
#print(benign)
'''
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


# Check if the path exists

# Create an instance of the dataset
dataset = ISISCDataset(
    benign_df=benign,
    malignant_df=malignant,
    images_dir=images,
    transform_benign=benign_aug,
    transform_malignant=malig_aug,
    augment_ratio=1.0
)

# Create a DataLoader to iterate through the dataset
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # Adjust batch size as needed

# Get a batch of images from the loader
for img1, img2, labels in data_loader:
    # We are using the first batch here; you could loop over more if needed
    break

# Plot the images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # Create a grid for 4 pairs of images

for i in range(4):
    # Convert img1 and img2 tensors back to images
    img1_np = tensor_to_image(img1[i])
    img2_np = tensor_to_image(img2[i])
    
    # Display img1 and img2 side by side
    axes[0, i].imshow(img1_np)
    axes[0, i].axis('off')
    axes[0, i].set_title(f'Image 1 (Label: {labels[i].item()})')

    axes[1, i].imshow(img2_np)
    axes[1, i].axis('off')
    axes[1, i].set_title(f'Image 2 (Label: {labels[i].item()})')

plt.tight_layout()
plt.show()
'''