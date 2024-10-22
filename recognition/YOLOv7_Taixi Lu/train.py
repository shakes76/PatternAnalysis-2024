
import torch
from dataset import get_dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(csv_file='data/train/ISIC-2017_Training_Part3_GroundTruth.csv',
                                root_dir='data/train', batch_size=8)
    # Print summary of the data
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of samples: {len(dataloader.dataset)}")
