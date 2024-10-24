import os
import kagglehub
from torchvision import datasets, transforms
from PIL import Image

class ISICDataset:
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        image_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.jpeg'):
                    image_files.append(os.path.join(root, file))
        return image_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def download_and_load_dataset():
    path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    print("Path to dataset files:", path)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = ISICDataset(path, transform=transform)

    return dataset

if __name__ == "__main__":
    dataset = download_and_load_dataset()
    print(f"Loaded {len(dataset)} images.")
