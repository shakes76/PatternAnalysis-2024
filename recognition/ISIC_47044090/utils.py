import os
import matplotlib.pyplot as plt
import re
from PIL import Image
import torch
import numpy as np
import matplotlib.patches as patches

original_filepath = "./datasets/ORIGINAL_ISIC"
modified_filepath = "./datasets/ISIC"

def scan_directory(partition):
    """
    Helper function that gathers all file codes in partition directory into a list

    Parameters:
        partition: test/train/val
    
    Return:
        list: list of all codes (e.g., [000111, 000124, ...] in the partition)
    """
    imgs = []
    for filename in os.scandir(f"{original_filepath}/{partition}/masks"):
        if filename.is_file():
            imgs.append(re.sub(r'\D', '', filename.path))
    
    return imgs


def get_newest_item(directory):
    """
    Get the full paths of all files and folders in the directory

    Parameters:
        directory (string): directory in which to search for newest file/folder
    
    Returns:
        string: path to newest file/folder
    """
    items = [os.path.join(directory, f) for f in os.listdir(directory)]

    if not items:
        return None  # Handle empty directories
    if './runs/detect/.DS_Store' in items:
        items.remove('./runs/detect/.DS_Store')
    
    newest_item = max(items, key=os.path.getmtime) # Find the newest item (file or folder) by modification time
    return newest_item


def iou_torch(true_bbox, pred_bbox):
    """
    Calcualtes the IoU for two bounding boxes (true and predicted)
    
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


def test_extraction(mask):
    """
    Helper function to test the extraction process by visualising a single sample and its bounding box
    """
    image = Image.open(f'{modified_filepath}/train/images/ISIC_{mask}.jpg')
    f = open(f"{modified_filepath}/train/labels/ISIC_{mask}.txt", "r")
    obj, x_center, y_center, width, height = np.array(f.read().split(" "))
    x_center, y_center, width, height = float(x_center)*512, float(y_center)*512, float(width)*512, float(height)*512
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle(((x_center) - (width)/2, (y_center) - (height)/2), (width), (height), linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title("Lesion detection example")
    plt.show()


def image_and_mask(sample):
    """
    Helper function to test the extraction process by visualising a single sample and its bounding box
    """
    fig, axs = plt.subplots(1,2)
    image = Image.open(f'{original_filepath}/train/images/ISIC_{sample}.jpg')
    mask = Image.open(f'{original_filepath}/train/masks/ISIC_{sample}_segmentation.png').convert('1')
    axs[0].imshow(image)
    axs[1].imshow(mask)

    axs[0].set_title("ISIC supplied image")
    axs[1].set_title("ISIC supplied mask")
    plt.show()

def image_and_mask(sample):
    """
    Helper function to test the extraction process by visualising a single sample and its bounding box
    """
    fig, axs = plt.subplots(1,2)
    image = Image.open(f'{original_filepath}/train/images/ISIC_{sample}.jpg')
    mask = Image.open(f'{modified_filepath}//train/images/ISIC_{sample}.jpg')
    axs[0].imshow(image)
    axs[1].imshow(mask)

    axs[0].set_title("original ISIC supplied image")
    axs[1].set_title("modified image for YOLO use")
    plt.show()


# image_and_mask("0000001")