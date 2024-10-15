from utils import *


class ADNIDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.img_paths = [os.path.join(path, img) for img in os.listdir(path)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img


# Apply transformations depending on grayscale or RGB data
if CHANNELS == 1:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(1),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((IMG_SIZE, IMG_SIZE))
    ])
elif CHANNELS == 3:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((IMG_SIZE, IMG_SIZE))
    ])

train_dataset = ADNIDataset(train_path, transform=transform)
test_dataset = ADNIDataset(test_path, transform=transform)

# Take a subset of the training data for validation
train_size = int(len(train_dataset) * 0.8)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Loaders
def get_loader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

train_loader = get_loader(train_dataset)
val_loader = get_loader(val_dataset)
test_loader = get_loader(test_dataset)


# Unit test on dataset and dataloader
def show_samples(dataloader, n_samples=16, columns=4):
    plt.figure(figsize=(16, 16))
    plt.title("Training Dataloader Sample")
    plt.axis("off")
    for i in range(n_samples):
        img = next(iter(dataloader))
        plt.subplot(int(n_samples / columns), columns, i + 1)
        plt.imshow(np.transpose(vutils.make_grid(img[0].to(DEVICE), normalize=True).cpu(), (1,2,0)))
    plt.show()

show_samples(train_loader)
