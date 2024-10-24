import cv2
import numpy as np
from PIL import Image
from numpy import asarray
from utils import *
import math

original_filepath = "./datasets/ORIGINAL_ISIC"
modified_filepath = "./datasets/ISIC"

def resize_mask(file, partition):
    """ 
    Re-sizes mask to (512, 512) by:
    1. Adds black letterbox to the smaller between x/y until x=y 
    2. Scales images down to (512, 512) (there can be 1 pixel stretching in 
        either direction due to rounding values, but this is negligible)
    
    then saves it in modified_file_path
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


def resize_image(file, partition):
    """ 
    Re-sizes image to (512, 512) by:
    1. Adds black letterbox to the smaller between x/y until x=y 
    2. Scales images down to (512, 512) (there can be 1 pixel stretching in 
        either direction due to rounding values, but this is negligible)
    
    then saves it in modified_file_path
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
    Extracts bounding box information from mask, creates .txt file

    mask must be resized first using resize_mask

    Parameters:
        file: code of sample
        partition: test/train/val
    """
    image = Image.open(f'{modified_filepath}/{partition}/masks/ISIC_{file}_segmentation.png')
    ys, xs = np.where(image == 255) # extract x and y coordinates of white pixels (=255)
    x_center = int(np.median(xs))
    width = np.max(xs) - np.min(xs) 
    y_center = int(np.median(ys))
    height = np.max(ys) - np.min(ys)
    with open(f"{modified_filepath}/{partition}/labels/ISIC_{file}.txt", "w") as file:
        file.write(f"0 {x_center/512} {y_center/512} {width/512} {height/512}") # write txt file


def process_dataset():
    """
    Runs resize_image, resize_mask and convert_mask_to_text for each image in directory for each partition.

    End result is fully processed dataset ready for Ultralytics
    """
    for partition in ["train", "test", "val"]:
        for file in scan_directory(partition):
            resize_image(file, partition)
            resize_mask(file, partition)
            convert_mask_to_txt(file, partition)


if __name__ == "__main__":
    process_dataset()