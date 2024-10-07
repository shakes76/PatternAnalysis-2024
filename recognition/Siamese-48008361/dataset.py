import os
import random
from collections import OrderedDict
from io import BytesIO
import mmap
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def preprocess_dataset(csv_file, img_dir, output_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create output directories
    benign_dir = os.path.join(output_dir, 'benign')
    malignant_dir = os.path.join(output_dir, 'malignant')
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_name = row['image_name'] + '.jpg'
        src_path = os.path.join(img_dir, img_name)
        
        if row['target'] == 0:  # Benign
            dst_path = os.path.join(benign_dir, img_name)
        else:  # Malignant
            dst_path = os.path.join(malignant_dir, img_name)
        
        shutil.copy(src_path, dst_path)

    print(f"Preprocessing complete. Images organized in {output_dir}")

class ISIC2020Dataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', split_ratio=0.8, cache_size=1000):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.cache_size = cache_size

        self.benign_dir = os.path.join(data_dir, 'benign')
        self.malignant_dir = os.path.join(data_dir, 'malignant')

        benign_images = self.get_image_paths(self.benign_dir, 0)
        malignant_images = self.get_image_paths(self.malignant_dir, 1)

        all_images = benign_images + malignant_images
        
        logging.info(f"Found {len(benign_images)} benign images and {len(malignant_images)} malignant images")

        if not all_images:
            raise ValueError("No images found in the dataset.")

        train_images, test_images = train_test_split(
            all_images,
            test_size=1 - split_ratio,
            stratify=[x[1] for x in all_images],
            random_state=42
        )

        self.images = train_images if mode == 'train' else test_images
        self.image_paths = [img_path for img_path, _ in self.images]
        self.labels = [label for _, label in self.images]
        logging.info(f"Using {len(self.images)} images for {mode}")

        self.class_to_indices = {
            0: [i for i, label in enumerate(self.labels) if label == 0],
            1: [i for i, label in enumerate(self.labels) if label == 1]
        }

        self.image_files = self.open_image_files()
        self.cache = OrderedDict()

    def get_image_paths(self, directory, label):
        image_paths = []
        for img in os.listdir(directory):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, img)
                if os.path.isfile(img_path) and os.path.getsize(img_path) > 0:
                    image_paths.append((img_path, label))
                else:
                    logging.warning(f"Skipping invalid or empty file: {img_path}")
        return image_paths

    def open_image_files(self):
        image_files = {}
        for img_path in tqdm(self.image_paths, desc=f"Opening {self.mode} image files"):
            try:
                f = open(img_path, 'rb')
                image_files[img_path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except IOError as e:
                logging.error(f"Error opening file {img_path}: {e}")
        return image_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        anchor = self.get_cached_image(img_path)
        positive = self.get_positive_image(label, exclude_img_path=img_path)
        negative = self.get_negative_image(label)

        return anchor, positive, negative, label

    def get_cached_image(self, img_path):
        if img_path in self.cache:
            self.cache.move_to_end(img_path)
            return self.cache[img_path]

        if img_path not in self.image_files:
            logging.warning(f"Image file not found: {img_path}. Using fallback image.")
            return self.get_fallback_image()

        try:
            image_data = self.image_files[img_path]
            image = Image.open(BytesIO(image_data.read()))
            image.load()
            image_data.seek(0)

            if self.transform:
                image = self.transform(image)

            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[img_path] = image

            return image
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            return self.get_fallback_image()

    def get_positive_image(self, label, exclude_img_path):
        matching_images = [img for img, lbl in zip(self.image_paths, self.labels) if lbl == label and img != exclude_img_path]
        if not matching_images:
            logging.warning(f"No positive images found for label {label}. Using anchor as positive.")
            return self.get_cached_image(exclude_img_path)
        
        img_path = random.choice(matching_images)
        return self.get_cached_image(img_path)

    def get_negative_image(self, label):
        opposite_label = 1 - label
        matching_images = [img for img, lbl in zip(self.image_paths, self.labels) if lbl == opposite_label]
        if not matching_images:
            logging.warning(f"No negative images found for label {label}. Using fallback image.")
            return self.get_fallback_image()
        
        img_path = random.choice(matching_images)
        return self.get_cached_image(img_path)

    def get_fallback_image(self):
        fallback = Image.new('RGB', (224, 224), color='gray')
        if self.transform:
            fallback = self.transform(fallback)
        return fallback

    def __del__(self):
        for mmap_file in self.image_files.values():
            mmap_file.close()

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.smallest_class_size = min(len(dataset.class_to_indices[0]), len(dataset.class_to_indices[1]))

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.batch_size // 2):
                batch.append(random.choice(self.dataset.class_to_indices[0]))  # Benign
                batch.append(random.choice(self.dataset.class_to_indices[1]))  # Malignant
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# this class was inspired by this git issue: https://github.com/huggingface/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238

class PrefetchLoader:
    def __init__(self, loader, mean, std):
        self.loader = loader
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input in self.loader:
            with torch.cuda.stream(stream):
                next_anchor, next_positive, next_negative, next_target = next_input
                next_anchor = next_anchor.cuda(non_blocking=True)
                next_positive = next_positive.cuda(non_blocking=True)
                next_negative = next_negative.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_anchor = next_anchor.float().sub_(self.mean).div_(self.std)
                next_positive = next_positive.float().sub_(self.mean).div_(self.std)
                next_negative = next_negative.float().sub_(self.mean).div_(self.std)

            if not first:
                yield anchor, positive, negative, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            anchor, positive, negative, target = next_anchor, next_positive, next_negative, next_target

        yield anchor, positive, negative, target

    def __len__(self):
        return len(self.loader)

def get_data_loaders(data_dir, batch_size=32, split_ratio=0.8, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ISIC2020Dataset(data_dir, transform=transform, mode='train', split_ratio=split_ratio)
    test_dataset = ISIC2020Dataset(data_dir, transform=transform, mode='test', split_ratio=split_ratio)

    train_sampler = BalancedBatchSampler(train_dataset, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_loader = PrefetchLoader(train_loader, mean, std)
    test_loader = PrefetchLoader(test_loader, mean, std)

    return train_loader, test_loader

if __name__ == '__main__':
    # Test the data loading
    csv_file = 'ISIC_2020_Training_GroundTruth_v2.csv'
    img_dir = 'data/ISIC_2020_Training_JPEG/train/'
    output_dir = 'preprocessed_data'
    
    preprocess_dataset(csv_file, img_dir, output_dir)
    data_dir = 'preprocessed_data'
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=32)
    
    for i, (anchor, positive, negative, labels) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"Anchor shape: {anchor.shape}")
        print(f"Positive shape: {positive.shape}")
        print(f"Negative shape: {negative.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        
        if i == 2:
            break