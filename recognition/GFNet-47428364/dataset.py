from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

""" Returns the train and test dataloaders for the ADNI dataset """
def get_dataloaders(batch_size=32, path="recognition/GFNet-47428364/AD_NC"):
    # Create transformer
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Generate datasets
    train_dataset = ImageFolder(root=path+"/train", transform=transform)
    test_dataset = ImageFolder(root=path+"/test", transform=transform)

    # Generate dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader