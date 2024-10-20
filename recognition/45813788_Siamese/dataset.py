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
    #normalise using ImageNet (pytorch defaults)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#augument dataset since theres only 584 malignant and 30000 non malig
malig_aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#augument dataset since theres only 584 malignant and 30000 non malig
benign_aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class ISICDataset(Dataset):
    def __init__(self, df, images_dir, transform_benign=None, transform_malignant=None, augment_ratio=1.0):
        self.df = df
        self.images_dir = images_dir
        self.transform_benign = transform_benign
        self.transform_malignant = transform_malignant
        self.augment_ratio = augment_ratio

        #list of image IDs
        self.image_ISIC = df['isic_id'].tolist()
        self.labels = df['target'].astype(int).tolist()

    def __len__(self):
        return len(self.image_ISIC)

    def apply_transform(self, img, label):
        transform = None
        #need to make sure its exclusively benign all else consider malig
        if label == 0:
            if self.transform_benign:
                transform = self.transform_benign
        else:
            if self.transform_malignant:
                transform=self.transform_malignant

        if transform is None:
            transform = default_aug

        img = transform(img)

        return img

    def __getitem__(self, index):
        img_id = self.image_ISIC[index]    
        label = self.labels[index]
       
        #get images
        img_path = os.path.join(self.images_dir,img_id + '.jpg')
        
        #normalise pixel values
        img = Image.open(img_path).convert('RGB')


        #transform the images
        img = self.apply_transform(img, label)
     
        
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