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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5,), (0.5,))
])

def get_loader(dataset, batch_size=16, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_dataset = ADNIDataset(train_path, transform=transform)
test_dataset = ADNIDataset(test_path, transform=transform)

# Take a subset of the training data for validation
train_size = int(len(train_dataset) * 0.8)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Loaders
train_loader = get_loader(train_dataset)
val_loader = get_loader(val_dataset, shuffle=False)
test_loader = get_loader(test_dataset, shuffle=False)
