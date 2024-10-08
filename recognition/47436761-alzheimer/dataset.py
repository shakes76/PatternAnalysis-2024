import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the paths
train_dir = '/home/groups/comp3710/ADNI/AD_NC/train'
test_dir = '/home/groups/comp3710/ADNI/AD_NC/test'
batch_size = 32

IMAGE_DIM = 240 

transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.ToTensor(), 
    # Using imagenet normalization https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Class-to-index mapping: {train_dataset.class_to_idx}")
