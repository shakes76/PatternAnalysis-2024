from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data paths
# train_dir = '/home/groups/comp3710/ADNI/AD_NC/train'
# test_dir = '/home/groups/comp3710/ADNI/AD_NC/test'

def get_data_loaders(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # # Fetch a single batch from the training loader
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    
    # print(f'Image batch shape: {images.shape}')  # [batch_size, 3, 224, 224]
    # print(f'Label batch shape: {labels.shape}')  # [batch_size]
    # print(f'First label in the batch: {labels[0]}') 
    return train_loader, test_loader



