from modules import yolo_model
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from utils import scan_directory

modified_filepath = "./datasets/ISIC"

def visualise_testcase(ax, file, true_bbox, pred_bbox):
    """
    Visualises a given testcase onto an axis

    Parameters:
        ax: axes on which to plot
        file: code of testcase
        true_bbox: the correct bounding box
        pred_bbox: bounding box predicted by model
    """
    image = Image.open(f'{modified_filepath}/test/images/ISIC_{file}.jpg')
    x_center1, y_center1, width1, height1 = true_bbox
    x_center2, y_center2, width2, height2 = pred_bbox
    ax.imshow(image)
    rect1 = patches.Rectangle(((x_center1) - (width1)/2, (y_center1) - (height1)/2), (width1), (height1), linewidth=1, edgecolor='r', facecolor='none', label='true')
    rect2 = patches.Rectangle(((x_center2) - (width2)/2, (y_center2) - (height2)/2), (width2), (height2), linewidth=1, edgecolor='blue', facecolor='none', label='pred')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    

def run_predict(run_number=-1, n_rows=3, partition='test'):
    """
    Runs the prediction and visualisation of a number of random samples

    Parameters:
        run_number: -1 if most recent, set to the training batch number (e.g. train4->4)
        n_rows: number of rows of 3 to visualise
        partition: partition from which to predict from 'train'/'test'/'val'
    """
    if run_number == -1:
        path = list(os.scandir(f"./runs/detect"))[-1].path + "/weights/best.pt"
    else:
        path = f"./runs/detect/train{run_number}/weights/best.pt"
    model = yolo_model(path)
    print(f"\bWeights used for test: {path}\n_______________________________________\n")
    
    fig, axs = plt.subplots(n_rows, 3)
    plt.suptitle("Random prediction examples")
    for i, (file, ax) in enumerate(zip(np.random.choice(scan_directory(partition), n_rows*3), axs.flat)): # first hundred to test
        results = model.predict(f"{modified_filepath}/{partition}/images/ISIC_{file}.jpg", imgsz=512, conf=0.1) 
        for result in results:
            pred_xywh = result.boxes.xywh
            true_xywh = [float(i)*512 for i in open(f"{modified_filepath}/{partition}/labels/ISIC_{file}.txt").read().split(" ")[1:]]
            visualise_testcase(ax, file, true_xywh, pred_xywh.tolist()[0])
            ax.set_title(f"{partition}_sample_{file}")
            if i==0:
                ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_predict()