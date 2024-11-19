import time
import torch.optim as optim
from matplotlib import pyplot as plt
from modules import *
from utils.general import non_max_suppression
from dataset import *
from PIL import Image
import numpy as np
import os


def train_yolo(model, dataloader, optimizer, loss_function, device, num_epochs=10):
    model.train()
    losses = []
    anchor_in_model = get_anchor(model)
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
                    ground_truth_image = ground_truth_image.resize(g_image_size)
                    ground_truth_arrays.append(np.array(ground_truth_image))
                    ground_truth_image.close()
                except FileNotFoundError:
                    print(f"Warning: Ground truth image for {img_name} not found.")
                    continue

            if len(ground_truth_arrays) == 0:
                continue

            grid_sizes = [80, 40, 20]
            targets = []
            for grid_size in grid_sizes:
                scale_targets = torch.stack([construct_yolo_target(
                    gt, grid_size=grid_size, target_class=labels[idx], anchors=anchor_in_model,
                    image_size=g_image_size, num_classes=g_num_classes, num_anchors=g_num_anchors)
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

    with torch.no_grad():
        for batch_idx, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            results = model(images)[0].cpu()
            results_NMS = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)

            del results
            torch.cuda.empty_cache()

            for i in range(len(results_NMS)):  # Iterate over each image in the batch
                try:
                    ground_truth_image = Image.open(os.path.join(ground_truth_path,
                                                                 f"{img_names[i]}_segmentation.png")).convert("L")
                    ground_truth_image = ground_truth_image.resize(g_image_size)
                    ground_truth_array = np.array(ground_truth_image)
                    ground_truth_image.close()
                except FileNotFoundError:
                    print(f"Warning: Ground truth image for {img_names[i]} not found.")
                    continue

                # print(f"img name: {img_names[i]}")
                gt_x_center, gt_y_center, gt_width, gt_height = get_YOLO_box(ground_truth_array)
                # print(f"gt_width: {gt_width}, gt_height: {gt_height}")
                gt_x1 = gt_x_center - (gt_width / 2)
                gt_y1 = gt_y_center - (gt_height / 2)
                gt_x2 = gt_x_center + (gt_width / 2)
                gt_y2 = gt_y_center + (gt_height / 2)

                # Extract predictions for the current image
                image_predictions = results_NMS[i]

                if len(image_predictions) > 0:
                    # check the one with max confidence for every image
                    max_conf_idx = torch.argmax(image_predictions[:, 4])
                    pred = image_predictions[max_conf_idx]
                    # confid = pred[4]
                    # if confid < 0.5:  # optional filter for Object confidence
                    #     continue

                    pred_box = pred[:4]
                    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = pred_box.detach().cpu().numpy()

                    # Calculate IoU with the ground truth
                    iou = calculate_iou([yolo_x1, yolo_y1, yolo_x2, yolo_y2],
                                        [gt_x1, gt_y1, gt_x2, gt_y2], len(all_ious) % 70 == 0)
                    all_ious.append(iou)
            del results_NMS
            torch.cuda.empty_cache()

    # Calculate average and maximum IoU
    avg_iou = np.mean(all_ious) if len(all_ious) > 0 else 0
    print(f"Average Intersection Over Union (IoU): {avg_iou:.4f}")
    max_iou = np.max(all_ious) if len(all_ious) > 0 else 0
    print(f"Maximum Intersection Over Union (IoU): {max_iou:.4f}")
    min_iou = np.min(all_ious) if len(all_ious) > 0 else 0
    print(f"Maximum Intersection Over Union (IoU): {min_iou:.4f}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(csv_file='data/train/ISIC-2017_Training_Part3_GroundTruth.csv',
                                root_dir='data/train/ISIC-2017_Training_Data', batch_size=4)
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Image dimensions: {dataloader.dataset[0][0].shape}")

    yolov7 = get_yolo_model(model_path='yolov7_training.pt', device=device)

    optimizer = optim.Adam(yolov7.parameters(), lr=0.001)
    loss_function = YOLOLoss(lambda_coord=100, lambda_noobj=0.5)
    train_yolo(yolov7, dataloader, optimizer, loss_function, device, num_epochs=15)

    # yolov7 = torch.load("trained model/yolov7_epoch_20.pt", map_location=device)
    # yolov7.to(device)
    validate_yolo(yolov7, dataloader, ground_truth_path='data/train/ISIC-2017_Training_Part1_GroundTruth',
                  device=device)
