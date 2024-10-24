import torch
from modules import *
from dataset import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.general import non_max_suppression


def predict_yolo(model, dataloader, ground_truth_path, device):
    model.eval()
    font = ImageFont.truetype("arial.ttf", 20)  # Load default font
    with torch.no_grad():
        for batch_idx, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            preds = model(images)[0].cpu()
            preds_NMS = non_max_suppression(preds, conf_thres=0.5, iou_thres=0.5)

            del preds
            torch.cuda.empty_cache()

            # gt_x1, gt_y1, gt_x2, gt_y2 = (0, 0, 0, 0)

            for i, pred in enumerate(preds_NMS):
                if pred is not None and len(pred) > 0:
                    image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
                    image = (image * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image)
                    draw = ImageDraw.Draw(image_pil)

                    # Load and draw ground truth contour
                    ground_truth_file = os.path.join(ground_truth_path, f"{img_names[i]}_segmentation.png")
                    try:
                        ground_truth_image = Image.open(ground_truth_file).convert("L")
                        ground_truth_image = ground_truth_image.resize(g_image_size)
                        ground_truth_array = np.array(ground_truth_image)
                        ground_truth_image.close()
                        gt_indices = np.argwhere(ground_truth_array > 0)
                        if len(gt_indices) > 0:
                            gt_x1, gt_y1 = gt_indices.min(axis=0)
                            gt_x2, gt_y2 = gt_indices.max(axis=0)
                            draw.rectangle((gt_x1, gt_y1, gt_x2, gt_y2), outline='green', width=2)
                        else:
                            continue
                    except FileNotFoundError:
                        print(f"Warning: Ground truth image for {img_names[i]} not found.")
                        continue


                    # check the one with max confidence for every image
                    max_conf_idx = torch.argmax(pred[:, 4])
                    pred = pred[max_conf_idx]
                    # confid = pred[4]
                    # if confid < 0.5:  # Object confidence
                    #     continue

                    pred_box = pred[:4]  # tx, ty, tw, th (center x, center y, width, height)
                    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = pred_box.detach().cpu().numpy()
                    draw.rectangle((yolo_x1, yolo_y1, yolo_x2, yolo_y2), outline='red', width=2)
                    draw.text((yolo_x1, yolo_y1), f"class: {pred[5]:.0f} confidence:{pred[4]:.2f}"
                              , fill='red', font=font)

                    # Calculate IoU
                    iou = calculate_iou([yolo_x1, yolo_y1, yolo_x2, yolo_y2],
                                        [gt_x1, gt_y1, gt_x2, gt_y2])

                    print(f"class: {pred[5]:.0f}  confidence:{pred[4]:.2f}  iou:{iou:.3f} \
                    gt box(green):{[gt_x1, gt_y1, gt_x2, gt_y2]} YOLO box(red):{[yolo_x1, yolo_y1, yolo_x2, yolo_y2]} ")

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
                                root_dir='data/test/ISIC-2017_Test_v2_Data', batch_size=1)
    yolov7 = torch.load("trained model/yolov7_epoch_1.pt", map_location=device)
    ground_truth_path = 'data/test/ISIC-2017_Test_v2_Part1_GroundTruth'
    yolov7.to(device)
    predict_yolo(yolov7, dataloader, ground_truth_path, device=device)
