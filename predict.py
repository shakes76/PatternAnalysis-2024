import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os.path
from modules import UNet, f1_score
from dataset import HipMRIDataset, DATASET_ROOT
from train import report_batch

if __name__ == '__main__':
    if not os.path.isfile('models/unet.pt'):
        raise RuntimeError('no saved model with which to predict any values')

    model: UNet = torch.load('models/unet.pt', weights_only=False)

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    transform = transforms.Normalize(mean=(0.5,), std=(0.5,))

    train_dataset = HipMRIDataset(DATASET_ROOT, train=True, transform=transform, device=device)
    test_dataset = HipMRIDataset(DATASET_ROOT, train=False, transform=transform, device=device)

    torch.manual_seed(0)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    scores_sum = 0.0
    scores_count = 0

    for batch, (images, segments) in enumerate(train_loader):
        predicted = model(images)
        f1 = f1_score(predicted, segments)
        scores_sum += sum(f1)
        scores_count += f1.numel()
        if batch % 50 == 0:
            report_batch(images, segments, predicted, batch, 'train')

    print(f'Mean F1 train score: {scores_sum / scores_count}')

    for batch, (images, segments) in enumerate(test_loader):
        predicted = model(images)
        f1 = f1_score(predicted, segments)
        scores_sum += sum(f1)
        scores_count += f1.numel()
        if batch % 50 == 0:
            report_batch(images, segments, predicted, batch, 'test')

    print(f'Mean F1 test score: {scores_sum / scores_count}')

