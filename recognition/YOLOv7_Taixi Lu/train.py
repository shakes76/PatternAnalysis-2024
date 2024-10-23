import time
import torch.optim as optim
from matplotlib import pyplot as plt
from modules import *
from dataset import get_dataloader
from PIL import Image
import numpy as np
import os


def train_yolo(model, dataloader, optimizer, loss_function, device, num_epochs=10):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]:")
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for batch_idx, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            # Load corresponding ground truth segmentation
            ground_truth_arrays = []
            for img_name in img_names:
                try:
                    ground_truth_image = Image.open(os.path.join('data/train/ISIC-2017_Training_Part1_GroundTruth',
                                                                 f"{img_name}_segmentation.png")).convert("L")
                    ground_truth_image = ground_truth_image.resize((images.shape[3], images.shape[2]))
                    ground_truth_arrays.append(np.array(ground_truth_image))
                except FileNotFoundError:
                    print(f"Warning: Ground truth image for {img_name} not found.")
                    continue

            if len(ground_truth_arrays) == 0:
                continue

            grid_sizes = [80, 40, 20]
            targets = []
            for grid_size in grid_sizes:
                scale_targets = torch.stack([construct_yolo_target(
                    gt, grid_size=grid_size, target_class=labels[idx],
                    image_size=(images.shape[2], images.shape[3]),
                    num_classes=g_num_classes, num_anchors=g_num_anchors)
                    for idx, gt in enumerate(ground_truth_arrays)])
                targets.append(scale_targets.to(device))

            optimizer.zero_grad()
            preds = model(images)
            loss = loss_function(preds, targets)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        time_left = epoch_duration * (num_epochs - (epoch + 1))
        print(f"    YOLO Training Loss: {avg_loss:.4f}")
        print(f"    Time taken for epoch: {epoch_duration:.2f} seconds")
        print(f"    Estimated time left: {time_left / 60:.2f} minutes")

        # Save the model after each epoch
        model_save_path = f"trained model/yolov7_epoch_{epoch + 1}.pt"
        torch.save(model, model_save_path)
        print(f"    Model saved to {model_save_path}")

    # Plot the training losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

def validate_yolo(model, dataloader, ground_truth_path, device):
    model.eval()
    all_ious = []
    for batch_idx, (images, labels, img_names) in enumerate(dataloader):
        images = images.to(device)
        results = model.forward(images)
        for i, res in enumerate(results):
            ground_truth_file = os.path.join(ground_truth_path, f"{img_names[i]}_segmentation.png")
            ground_truth_image = Image.open(ground_truth_file).convert("L")
            ground_truth_array = np.array(ground_truth_image)

            if res is not None and len(res) > 0:
                iou = calculate_iou(res, ground_truth_array)
                all_ious.append(iou)

    avg_iou = np.mean(all_ious) if len(all_ious) > 0 else 0
    max_iou = np.max(all_ious) if len(all_ious) > 0 else 0
    print(f"Average Intersection Over Union (IoU): {avg_iou:.4f}")
    print(f"Maximum Intersection Over Union (IoU): {max_iou:.4f}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(csv_file='data/train/ISIC-2017_Training_Part3_GroundTruth.csv', root_dir='data/train',
                                batch_size=8)
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Image dimensions: {dataloader.dataset[0][0].shape}")

    yolov7 = get_yolo_model(model_path='yolov7_training.pt', device=device)
    # print(f"Model loaded: \n{yolov7}")

    optimizer = optim.Adam(yolov7.parameters(), lr=0.001)
    loss_function = YOLOLoss()
    train_yolo(yolov7, dataloader, optimizer, loss_function, device, num_epochs=10)

    validate_yolo(yolov7, dataloader, ground_truth_path='data/train/ISIC-2017_Training_Part1_GroundTruth',
                  device=device)

