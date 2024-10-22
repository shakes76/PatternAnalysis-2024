import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import YOLOv7Model, ClassificationYOLO
from dataset import get_dataloader



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(csv_file='data/train/ISIC-2017_Training_Part3_GroundTruth.csv',
                                root_dir='data/train/', batch_size=8)
    # Print summary of the data
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of samples: {len(dataloader.dataset)}")

    yolov7 = YOLOv7Model(model_path='yolov7.pt', device=device)
    classifier = ClassificationYOLO(input_dim=640, num_classes=2).to(device)

