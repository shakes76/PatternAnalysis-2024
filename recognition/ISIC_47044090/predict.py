from modules import yolo_model
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from utils import scan_directory

modified_filepath = "./datasets/ISIC"

def visualise_single_testcase(file: str, run_number=-1, partition='test'):
    """
    Runs the prediction and visualisation of a number of random samples

    Parameters:
        file(str): testcase CODE as a string e.g., "0000001"
        run_number: -1 if most recent, set to the training batch number (e.g. train4->4)
        partition: partition from which to predict from 'train'/'test'/'val'
    """
    if run_number == -1: # most recent weights
        path = list(os.scandir(f"./runs/detect"))[-1].path + "/weights/best.pt"
    else: # specific weights
        path = f"./runs/detect/train{run_number}/weights/best.pt"

    model = yolo_model(path)
    print(f"\bWeights used for test: {path}\n_______________________________________\n")
    
    fig, ax = plt.subplots()
    ax.set_title(f"Prediction of {partition} sample {file} ")
    image = Image.open(f'{modified_filepath}/test/images/ISIC_{file}.jpg') # get image
    ax.imshow(image)
    results = model.predict(f"{modified_filepath}/{partition}/images/ISIC_{file}.jpg", imgsz=512, conf=0.1) 
    for result in results: 
        pred_xywh = result.boxes.xywh # extracts bounding box information
        true_xywh = [float(i)*512 for i in open(f"{modified_filepath}/{partition}/labels/ISIC_{file}.txt").read().split(" ")[1:]]
        x_center1, y_center1, width1, height1 = true_xywh

        if pred_xywh.size() != torch.Size([0, 4]): # if was at least 1 detection
            pred_xywh = pred_xywh.tolist()[0] # always only look at most confident detection
            x_center2, y_center2, width2, height2 = pred_xywh
            rect2 = patches.Rectangle(((x_center2) - (width2)/2, (y_center2) - (height2)/2), (width2), (height2), linewidth=1, edgecolor='blue', facecolor='none', label='pred')
            ax.add_patch(rect2)

        rect1 = patches.Rectangle(((x_center1) - (width1)/2, (y_center1) - (height1)/2), (width1), (height1), linewidth=1, edgecolor='r', facecolor='none', label='true')
        ax.add_patch(rect1)


    ax.legend() # legend on just the first one
    plt.show()


def visualise_multiple_testcase(ax, file, true_bbox, pred_bbox):
    """
    Visualises a given testcase onto an axis

    Parameters:
        ax: axes on which to plot
        file: code of testcase
        true_bbox: the correct bounding box
        pred_bbox: bounding box predicted by model
    """
    image = Image.open(f'{modified_filepath}/test/images/ISIC_{file}.jpg') # get image
    x_center1, y_center1, width1, height1 = true_bbox
    if pred_bbox != None: # if no detection, don't try to plot
        x_center2, y_center2, width2, height2 = pred_bbox
        rect2 = patches.Rectangle(((x_center2) - (width2)/2, (y_center2) - (height2)/2), (width2), (height2), linewidth=1, edgecolor='blue', facecolor='none', label='pred')
        ax.add_patch(rect2)
    ax.imshow(image)
    rect1 = patches.Rectangle(((x_center1) - (width1)/2, (y_center1) - (height1)/2), (width1), (height1), linewidth=1, edgecolor='r', facecolor='none', label='true')
    ax.add_patch(rect1)
    
    
def run_predict(run_number=-1, n_rows=3, partition='test'):
    """
    Runs the prediction and visualisation of a number of random samples

    Parameters:
        testcases: list (if not supplied, fully random) of testcase CODES as strings (e.g., ["0000011", ...])
                    keep in multiples of 4
        run_number: -1 if most recent, set to the training batch number (e.g. train4->4)
        n_rows: number of rows of 4 to visualise
        partition: partition from which to predict from 'train'/'test'/'val'
    """
    if run_number == -1: # most recent weights
        path = list(os.scandir(f"./runs/detect"))[-1].path + "/weights/best.pt"
    else: # specific weights
        path = f"./runs/detect/train{run_number}/weights/best.pt"

    model = yolo_model(path)
    print(f"\bWeights used for test: {path}\n_______________________________________\n")
    
    fig, axs = plt.subplots(n_rows, 4)
    plt.suptitle("Random prediction examples")
    for i, (file, ax) in enumerate(zip(np.random.choice(scan_directory(partition), n_rows*4), axs.flat)): 
        results = model.predict(f"{modified_filepath}/{partition}/images/ISIC_{file}.jpg", imgsz=512, conf=0.1) 
        for result in results: 
            pred_xywh = result.boxes.xywh # extracts bounding box information
            true_xywh = [float(i)*512 for i in open(f"{modified_filepath}/{partition}/labels/ISIC_{file}.txt").read().split(" ")[1:]]
            
            if pred_xywh.size() != torch.Size([0, 4]):
                pred_xywh = pred_xywh.tolist()[0] # if no detection
            else:
                pred_xywh = None

            visualise_multiple_testcase(ax, file, true_xywh, pred_xywh) # plot to axis
            ax.set_title(f"{partition}_sample_{file}")
            if i==0:
                ax.legend() # legend on just the first one

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_predict()
    # visualise_single_testcase("0012092")