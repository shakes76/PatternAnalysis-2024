from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Variables to store total sum and total squared sum

class GFNetDataloader():
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self._mean = 0.0
        self._std = 0.0
        self._total_images = 0
        self.train_loader = None
        self.test_loader = None
        self.img_size = None

    def load(self, img_size=None):
        transform_complete = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        # dataset = datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/train', transform=transform_complete)
        dataset = datasets.ImageFolder(root='./AD_NC/train', transform=transform_complete)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        for images, _ in loader:
            # Compute the mean and std per batch
            batch_samples = images.size(0)  # Get the number of images in the batch
            images = images.view(batch_samples, images.size(1), -1)  # Flatten the image
            self._mean += images.mean(2).sum(0)  # Sum mean across batch
            self._std += images.std(2).sum(0)    # Sum std across batch
            self._total_images += batch_samples

        # Final mean and std calculations
        self._mean /= self._total_images
        self._std /= self._total_images

        val_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(240),
                transforms.Pad((0, 8), fill=0),
                transforms.Normalize(mean=self._mean, std=self._std)
                ])

        if not img_size:
            _transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(240),
                transforms.Pad((0, 8), fill=0),
                transforms.RandomHorizontalFlip(),              # Flip horizontally
                transforms.RandomVerticalFlip(p=0.5),           # Flip vertically with 50% probability
                transforms.RandomRotation(30),                  # Random rotation within ±30 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness/contrast
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
                transforms.RandomCrop(256, padding=8, padding_mode='reflect'),
                transforms.Normalize(mean=self._mean, std=self._std),  # Use computed mean and std
                ])
            self.img_size = 256
        else:
            exit(1)
            _transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(256, padding=8, padding_mode='reflect'),
                transforms.Normalize(mean=self._mean, std=self._std)  # Use computed mean and std
            ])
            self.img_size = img_size


        train_images = datasets.ImageFolder(root='./AD_NC/train', transform=_transform)
        test_images = datasets.ImageFolder(root='./AD_NC/test', transform=val_trans)
        # train_images = datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/train', transform=_transform)
        # test_images = datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/test', transform=val_trans)

        self.train_loader = DataLoader(train_images, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_images, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        return self.train_loader, self.test_loader

    def get_meta(self):
        return {"total_images": self._total_images,
                "mean": self._mean,
                "std": self._std,
                "img_size": self.img_size,
                "channels": 1,
                } 

# if __name__ == '__main__':
#     print('Loading Dataset')
#     data = GFNetDataloader()
#     data.load()
#     train, test = data.get_data()
#     data_iter = iter(train)
#     images, labels = next(data_iter)

#     # Undo normalization for display (normalize back to [0, 1] range)
#     unnormalize = transforms.Compose([
#         transforms.Normalize(mean=[-data._mean / data._std],
#                              std=[1.0 / data._std]),
#         ToPILImage()
#     ])

#     # Display the first image in the batch
#     image = unnormalize(images[0])  # Unnormalize and convert to PIL
#     plt.imshow(image, cmap='gray')
#     plt.show()
#     for thing in train:
#         print(thing[0].shape)
#         exit()
