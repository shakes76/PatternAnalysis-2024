import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules import UNet, f1_score
from dataset import HipMRIDataset, DATASET_ROOT
import matplotlib.pyplot as plt
import os, os.path

def report_batch(images: torch.Tensor, segmented: torch.Tensor, predicted: torch.Tensor, batch: int, name: str = 'batch') -> torch.Tensor:
    if images.shape[0] < 20 or segmented.shape[0] < 20 or predicted.shape[0] < 20:
        raise ValueError('cannot have less than 20 values to report a batch')

    fig, axes = plt.subplots(4, 5)
    fig.suptitle('Training Images')
    fig.set_size_inches(12, 8)
    fig.set_dpi(1000)

    for i, axis in enumerate(axes.ravel()):
        axis.set_title(f'{i + 1}th image')
        axis.imshow(images[i].squeeze().cpu().detach().transpose(0, 1).flip(0), cmap='gist_gray')
        axis.set_xticks([])
        axis.set_yticks([])

    plt.savefig(f'{name}_{batch}_images.png')
    plt.clf()

    fig, axes = plt.subplots(4, 5)
    fig.suptitle('Prostate Segments')
    fig.set_size_inches(12, 8)
    fig.set_dpi(1000)

    for i, axis in enumerate(axes.ravel()):
        axis.set_title(f'{i + 1}th image')
        axis.imshow(segmented[i].squeeze().cpu().detach().transpose(0, 1).flip(0), cmap='gist_gray')
        axis.set_xticks([])
        axis.set_yticks([])

    plt.savefig(f'{name}_{batch}_segmented.png')
    plt.clf()

    fig, axes = plt.subplots(4, 5)
    fig.suptitle('Predicted Segments')
    fig.set_size_inches(12, 8)
    fig.set_dpi(1000)

    for i, axis in enumerate(axes.ravel()):
        axis.set_title(f'{i + 1}th image')
        axis.imshow(predicted[i].squeeze().cpu().detach().transpose(0, 1).flip(0), cmap='gist_gray')
        axis.set_xticks([])
        axis.set_yticks([])

    plt.savefig(f'{name}_{batch}_predicted.png')
    plt.clf()

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Normalize(mean=(0.5,), std=(0.5,))
    train_dataset = HipMRIDataset(DATASET_ROOT, train=True, concise=False, device=device, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = UNet().to(device)

    criterion = nn.BCELoss()
    optimiser = optim.SGD(model.parameters(), lr=0.01)

    for batch, (images, segments) in enumerate(train_loader):
        images: torch.Tensor
        segments: torch.Tensor

        optimiser.zero_grad()
        output: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(output, segments)

        loss.backward()
        optimiser.step()

        print(f'Batch {batch + 1:3d} loss: {loss.item():.3f}')

        if batch % 50 == 0:
            report_batch(images, segments, output, batch)

    if not os.path.isdir('models/'):
        os.mkdir('models/')

    print('saving model to models/unet.pt')
    torch.save(model, 'models/unet.pt')

    loader = DataLoader(train_dataset, batch_size=20, shuffle=False)
    sample_images, sample_segments = next(iter(loader))
    predicted: torch.Tensor = model(sample_images)
    print(f'F1 Scores:')
    print(f1_score(predicted, sample_segments).tolist())

    report_batch(sample_images, sample_segments, predicted, 1_000)

    print('Done Learning')
