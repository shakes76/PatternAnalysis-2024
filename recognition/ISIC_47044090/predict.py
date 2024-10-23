from modules import yolo_model
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from dataset import scan_directory, get_newest_item, iou_torch

modified_filepath = "./datasets/ISIC"

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