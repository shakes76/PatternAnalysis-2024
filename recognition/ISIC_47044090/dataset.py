import cv2
import numpy as np
from PIL import Image
from numpy import asarray
import math
from matplotlib import cm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

original_filepath = "data/ORIGINAL_ISIC"
modified_filepath = "data/ISIC"

def resize_image(file, partition):
    """ 
    Re-sizes image to (512, 512)

    1. Adds black letterbox to the smaller between x/y until x=y 
    2. Scales images down to (512, 512) (there can be 1 pixel stretching in 
        either direction due to rounding values, but this is negligible)
    """
    image = Image.open(f'{original_filepath}/{partition}/labels/ISIC_{file}_segmentation.png')    
    width, height = image.size
    if width >= height:
        image = cv2.copyMakeBorder(asarray(image), math.ceil((width-height)/2), math.floor((width-height)/2), 0, 0, cv2.BORDER_CONSTANT,value=0)
    else: # height > width
        image = cv2.copyMakeBorder(asarray(image), 0, 0, math.ceil((height-width)/2), math.floor((height-width)/2), 0, 0, cv2.BORDER_CONSTANT,value=0)
    image = cv2.resize(image, (512, 512)) # CAN be minor rescaling, fixes fluxuations of 1 in either direction due to rounding
    im = Image.fromarray(image)
    im.save(f"{modified_filepath}/{partition}/labels/ISIC_{file}_segmentation.png")

def resize_mask(file, partition):
    """ 
    Re-sizes mask to (512, 512)

    1. Adds black letterbox to the smaller between x/y until x=y 
    2. Scales images down to (512, 512) (there can be 1 pixel stretching in 
        either direction due to rounding values, but this is negligible)
    """
    image = Image.open(f'{original_filepath}/{partition}/images/ISIC_{file}.jpg')    
    width, height = image.size
    if width >= height:
        image = cv2.copyMakeBorder(asarray(image), math.ceil((width-height)/2), math.floor((width-height)/2), 0, 0, cv2.BORDER_CONSTANT,value=0)
    else: # height > width
        image = cv2.copyMakeBorder(asarray(image), 0, 0, math.ceil((height-width)/2), math.floor((height-width)/2), 0, 0, cv2.BORDER_CONSTANT,value=0)
    image = cv2.resize(image, (512, 512)) # CAN be minor rescaling, fixes fluxuations of 1 in either direction due to rounding
    im = Image.fromarray(image)
    im.save(f"{modified_filepath}/{partition}/images/ISIC_{file}.jpg")


def convert_mask_to_txt(file, partition):
    """
    Extracts bounding box information from mask, creates txt file 

    mask must be resized first using resize_mask
    """
    image = Image.open(f'{modified_filepath}/{partition}/masks/ISIC_{file}_segmentation.png')
    ys, xs = np.where(image == 255) # extract x and y coordinates of white pixels (=255)
    x_center = int(np.median(xs))
    width = np.max(xs) - np.min(xs) 
    y_center = int(np.median(ys))
    height = np.max(ys) - np.min(ys)
    with open(f"{modified_filepath}/{partition}/labels/ISIC_{file}.txt", "w") as file:
        file.write(f"0 {x_center/512} {y_center/512} {width/512} {height/512}") # write txt file


def scan_directory(partition):
    """
    Gathers all labels in directory into a list
    """
    imgs = []
    for filename in os.scandir(f"{original_filepath}/{partition}/labels"):
        if filename.is_file():
            imgs.append(str(filename.path[32+len(partition):-17]))
    if "" in imgs:
        imgs.remove("")
    return imgs


def process_dataset():
    """
    Runs resize and mask_to_text for each image in directory for a single partition
    """
    for partition in ["train", "test", "val"]:
        for file in scan_directory(partition):
            resize_image(file, partition)
            resize_mask(file, partition)
            convert_mask_to_txt(file, partition)


def test_extraction(mask):
    """
    Helper function to test the extraction process by visualising a single sample and its bounding box
    """
    image = Image.open(f'{modified_filepath}/train/imgs/ISIC_{mask}.jpg')
    f = open(f"{modified_filepath}/train/txt/ISIC_{mask}.txt", "r")
    obj, x_center, y_center, width, height = np.array(f.read().split(" "))
    x_center, y_center, width, height = float(x_center)*512, float(y_center)*512, float(width)*512, float(height)*512
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle(((x_center) - (width)/2, (y_center) - (height)/2), (width), (height), linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    process_dataset()