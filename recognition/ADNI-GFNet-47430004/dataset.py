import torchvision.transforms as tf
from torchvision.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
    
    def load_data(self):
        images = []
        labels = []

        label_names = {"AD": 1, "NC": 0}
        sub_directory = "train" if self.train else "test"

        for label in label_names.keys():
            label_directory = os.path.join(self.root_dir, sub_directory, label)
            for img_name in os.listdir(label_directory):
                img_path = os.path.join(label_directory, img_name)
                images.append(img_path)
                labels.append(label_names[label])
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('L')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(data_dir, batch_size=32, crop_size=224, image_size=224):
    transform = tf.Compose([
        tf.Grayscale(num_output_channels=1),
        tf.CenterCrop(crop_size),
        tf.Resize((image_size, image_size)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.5],
                     std=[0.5]),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip()
    ])
    
    # Create datasets
    train_dataset = ADNIDataset(root_dir=data_dir, transform=transform, train=True)
    test_dataset = ADNIDataset(root_dir=data_dir, transform=transform, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader