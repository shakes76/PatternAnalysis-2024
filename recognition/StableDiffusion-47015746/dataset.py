import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_data(root_Train, root_Test):
    # Root directory for dataset
    data_Train = root_Train
    data_Test = root_Test


    # Spatial size of training images
    image_size = 64

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 4


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = dset.ImageFolder(root= data_Train, transform=transform)
    test_dataset = dset.ImageFolder(root= data_Test, transform=transform)
    combined_dataset = data.ConcatDataset([train_dataset, test_dataset])


    dataloader = data.DataLoader(combined_dataset, batch_size, shuffle=True, num_workers= workers)
    
    return dataloader


def test_load_data():
    # Replace with the path to your dataset
      # Ensure this path points to your image dataset
    data_train = "C:/Users/msi/Desktop/AD_NC/train" 
    data_test = "C:/Users/msi/Desktop/AD_NC/test" 
    # Test load_data function
    dataloader = load_data(data_train, data_test)

    # Check if DataLoader loads data and inspect the first batch
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        images = images*0.5 + 0.5
        # Visualize one image from the batch
        img = transforms.ToPILImage()(images[0])
        img.show()  # This will open the first image in the batch for inspection
        
        # Break after the first batch (for testing purposes)
        break

#if __name__ == "__main__":
    #test_load_data()

