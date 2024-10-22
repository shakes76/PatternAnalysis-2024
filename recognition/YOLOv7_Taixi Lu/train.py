import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import YOLOv7Model, ClassificationYOLO
from dataset import get_dataloader
from PIL import Image
import numpy as np
import os

def calculate_iou(yolo_output, ground_truth):
    # Calculate Intersection Over Union (IoU) between YOLO output and ground truth mask
    ious = []
    for yolo_box in yolo_output:
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box[:4]
        gt_indices = np.argwhere(ground_truth > 0)
        if len(gt_indices) == 0:
            continue
        gt_x1, gt_y1 = gt_indices.min(axis=0)
        gt_x2, gt_y2 = gt_indices.max(axis=0)

        inter_x1 = max(yolo_x1, gt_x1)
        inter_y1 = max(yolo_y1, gt_y1)
        inter_x2 = min(yolo_x2, gt_x2)
        inter_y2 = min(yolo_y2, gt_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

        union_area = yolo_area + gt_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        ious.append(iou)

    return np.mean(ious) if len(ious) > 0 else 0


def validate_yolo(model, dataloader, ground_truth_path, device):
    all_features = []
    all_labels = []
    all_ious = []
    for batch_idx, (images, labels, img_names) in enumerate(dataloader):
        images = images.to(device)
        results = model.forward(images)
        for i, res in enumerate(results):
            ground_truth_file = os.path.join(ground_truth_path, f"{img_names[i]}_segmentation.png")
            ground_truth_image = Image.open(ground_truth_file).convert("L")
            ground_truth_array = np.array(ground_truth_image)

            if res is not None and len(res) > 0:
                features = torch.mean(res[:, :4], dim=0).view(1, -1).to(device)
                all_features.append(features)
                all_labels.append(labels[i].clone().detach().to(device).long())
                iou = calculate_iou(res, ground_truth_array)
                all_ious.append(iou)

    avg_iou = np.mean(all_ious) if len(all_ious) > 0 else 0
    max_iou = np.max(all_ious) if len(all_ious) > 0 else 0
    print(f"Average Intersection Over Union (IoU): {avg_iou:.4f}")
    print(f"Maximum Intersection Over Union (IoU): {max_iou:.4f}")
    return torch.cat(all_features, dim=1), torch.cat(all_labels, dim=1)


def train_classifier(classifier, features, labels, num_epochs=10, device='cpu'):
    classifier.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    train_losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        avg_loss = loss.item()
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(csv_file='data/train/ISIC-2017_Training_Part3_GroundTruth.csv', root_dir='data/train', batch_size=8)
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Image dimensions: {dataloader.dataset[0][0].shape}")

    yolov7 = YOLOv7Model(model_path='yolov7.pt', device=device)
    features, labels = validate_yolo(yolov7, dataloader, ground_truth_path='data/train/ISIC-2017_Training_Part1_GroundTruth', device=device)

    classifier = ClassificationYOLO(input_dim=features.shape[1], num_classes=3).to(device)
    train_classifier(classifier, features, labels, num_epochs=10, device=device)