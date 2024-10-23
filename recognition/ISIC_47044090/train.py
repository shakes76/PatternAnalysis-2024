from modules import yolo_model
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from dataset import scan_directory

yaml_path = "./datasets" # path TO (but before yaml): also want processed test/train/val here
original_filepath = "./datasets/ORIGINAL_ISIC"
modified_filepath = "./datasets/ISIC"


def run_train(yaml_path):
    model = yolo_model("yolov7.pt") # initial weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = model.train(batch=32, device=device, data=f"{yaml_path}/isic.yaml", epochs=200, imgsz=512)


def visualise_testcase(file, true_bbox, pred_bbox):
    """
    Visualises 
    """
    image = Image.open(f'{modified_filepath}/test/images/ISIC_{file}.jpg')
    x_center1, y_center1, width1, height1 = true_bbox
    x_center2, y_center2, width2, height2 = pred_bbox
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect1 = patches.Rectangle(((x_center1) - (width1)/2, (y_center1) - (height1)/2), (width1), (height1), linewidth=1, edgecolor='r', facecolor='none', label='true_bbox')
    rect2 = patches.Rectangle(((x_center2) - (width2)/2, (y_center2) - (height2)/2), (width2), (height2), linewidth=1, edgecolor='blue', facecolor='none', label='test_bbox')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.legend()
    plt.show()


def iou_torch(true_bbox, pred_bbox):
    """
    Parameters:
        true_bbox=torch([[center x, center y, width, height]]) 
        pred_bbox=torch([[center x, center y, width, height]]) 

    Returns:
        float=IoU (0-1) of the boxes
    """
    print(pred_bbox)
    print(true_bbox)
    x1, y1, w1, h1 = true_bbox[0] # unpack inputs
    x2, y2, w2, h2 = pred_bbox[0] 
    
    true_bbox_x1, true_bbox_y1 = x1 - w1 / 2, y1 - h1 / 2 # bbox coordinates
    true_bbox_x2, true_bbox_y2 = x1 + w1 / 2, y1 + h1 / 2
    pred_bbox_x1, pred_bbox_y1 = x2 - w2 / 2, y2 - h2 / 2
    pred_bbox_x2, pred_bbox_y2 = x2 + w2 / 2, y2 + h2 / 2

    # coordinates of intersection
    inter_x1, inter_y1 = torch.max(true_bbox_x1, pred_bbox_x1), torch.max(true_bbox_y1, pred_bbox_y1)
    inter_x2, inter_y2 = torch.min(true_bbox_x2, pred_bbox_x2), torch.min(true_bbox_y2, pred_bbox_y2)
    
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height # intersection area

    true_bbox_area = w1 * h1 # area of both boxes
    pred_bbox_area = w2 * h2
    union_area = true_bbox_area + pred_bbox_area - inter_area # union area

    iou_value = inter_area / union_area if union_area > 0 else 0 # compute IoU (avoid division by zero)
    return np.round(iou_value.tolist(), 3)


def run_test(run_number=-1):
    """
    Parameters:
        trained_model: path to the trained model .pt file
    
    to run test, just run inference on test set - get the bounding boxes and compare?
    makes the most sense to me
    """
    if run_number == -1:
        path = list(os.scandir(f"./runs/detect"))[-1].path + "/weights/best.pt"
    else:
        path = f"./runs/detect/train{run_number}/weights/best.pt"
    model = yolo_model(path)
    print(f"\bWeights used for test: {path}\n_______________________________________\n")
    
    ious = []
    for file in scan_directory("test"): # first hundred to test
        results = model([f"{modified_filepath}/test/images/ISIC_{file}.jpg"]) 
        for result in results:
            pred_bbox = result.boxes.xywh
            if pred_bbox.size()==torch.Size([0, 4]):
                iou = 0
            else:
                if pred_bbox.size() != torch.Size([1, 4]):
                    pred_bbox = [pred_bbox[0]] # takes the most confident classification if multiple exist
                pred_bbox = pred_bbox 
                true_xywh = [float(i)*512 for i in open(f"{modified_filepath}/test/labels/ISIC_{file}.txt").read().split(" ")[1:]]
                # visualise_testcase(file, label_xywh, pred_bbox.tolist()[0])
                true_xywh=torch.tensor([true_xywh])
                iou = iou_torch(pred_bbox, true_xywh)

            print(iou)
            ious.append(iou)

    print(np.array(ious).mean())


if __name__ == '__main__':
    # run_train(yaml_path)
    run_test()