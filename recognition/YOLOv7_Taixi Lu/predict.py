import torch
from modules import *
from dataset import get_dataloader
from PIL import Image, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt


def predict_yolo(model, dataloader, ground_truth_path, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            preds = model(images)

            for i, pred in enumerate(preds):
                image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
                image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(image_pil)

                # Draw predictions on the image
                if pred is not None and len(pred) > 0:
                    for p in pred:
                        x1, y1, x2, y2, conf, cls = p[:6]
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                        draw.text((x1, y1), f"{cls:.0f}:{conf:.2f}", fill='red')

                # Load and draw ground truth contour
                ground_truth_file = os.path.join(ground_truth_path, f"{img_names[i]}_segmentation.png")
                try:
                    ground_truth_image = Image.open(ground_truth_file).convert("L")
                    ground_truth_array = np.array(ground_truth_image)
                    gt_indices = np.argwhere(ground_truth_array > 0)
                    if len(gt_indices) > 0:
                        gt_x1, gt_y1 = gt_indices.min(axis=0)
                        gt_x2, gt_y2 = gt_indices.max(axis=0)
                        draw.rectangle([gt_x1, gt_y1, gt_x2, gt_y2], outline='green', width=2)
                except FileNotFoundError:
                    print(f"Warning: Ground truth image for {img_names[i]} not found.")
                    continue

                # Calculate IoU
                iou = calculate_iou(pred, ground_truth_array) if len(pred) > 0 else 0

                # Plot and save the result
                plt.figure()
                plt.imshow(image_pil)
                plt.title(f"Prediction for {img_names[i]} | IoU: {iou:.4f}")
                plt.axis('off')
                plt.savefig(f"data/test/predictions/{img_names[i]}_prediction.png")
                plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(csv_file='data/test/ISIC-2017_Test_v2_Part3_GroundTruth.csv',
                                root_dir='data/test', batch_size=1)
    yolov7 = torch.load("trained model/yolov7_epoch_1.pt", map_location=device)
    ground_truth_path = 'data/test/ISIC-2017_Test_v2_Part1_GroundTruth'
    yolov7.to(device)
    predict_yolo(yolov7, dataloader, ground_truth_path, device=device)
