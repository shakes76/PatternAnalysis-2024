import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import CNN, SiameseNetwork, ContrastiveLoss
from dataset import ISICDataset, SiameseDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_siamese_network(dataset, transform=None):
    train_dataloader = DataLoader(SiameseDataset(dataset, transform=transform), batch_size=16, shuffle=True)
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # print number of iterations this loop will have
    print(f'Number of iterations: {len(train_dataloader)}')

    for i, (img1, img2, labels) in enumerate(train_dataloader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()

        similarity = net(img1, img2)

        loss = criterion(*similarity, labels.float())

        loss.backward()
        optimizer.step()

        print(f'Iteration: {i}, Loss: {loss.item()}')

if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    dataset_image_path = os.path.join(dataset_path, "train-image/image")
    meta_data_path = os.path.join(dataset_path, "train-metadata.csv")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    isic_dataset = ISICDataset(dataset_path=dataset_image_path, metadata_path=meta_data_path)

    train_siamese_network(dataset=isic_dataset, transform=transform)