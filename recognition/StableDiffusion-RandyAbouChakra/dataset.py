import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_data(root):
    # Root directory for dataset
    dataroot = root


    # Spatial size of training images
    image_size = 64

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Dataset and DataLoader
    dataset = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    return dataloader


def test_load_data(root):
    # Replace with the path to your dataset
    dataset_root = root  # Ensure this path points to your image dataset

    # Test load_data function
    dataloader = load_data(dataset_root)

    # Check if DataLoader loads data and inspect the first batch
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")

        # Visualize one image from the batch
        img = transforms.ToPILImage()(images[0])
        img.show()  # This will open the first image in the batch for inspection
        
        # Break after the first batch (for testing purposes)
        break

if __name__ == "__main__":
    test_load_data("/home/groups/comp3710/ADNI")

