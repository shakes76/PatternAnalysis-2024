import cv2
import numpy as np
from PIL import Image
from numpy import asarray
import math
from matplotlib import cm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def resize_img(mask, partition):
    """ 
    Adds black letterbox until x=y
    Scales then scales images down to 512, 512 (there can be 1 pixel stretching in 
    either direction due to rounding values, but this is negligible)
    """
    image = Image.open(f'data/ORIGINAL_ISIC/{partition}/labels/ISIC_{mask}_segmentation.png')    
    width, height = image.size
    if width >= height:
        image = cv2.copyMakeBorder(asarray(image), math.ceil((width-height)/2), math.floor((width-height)/2), 0, 0, cv2.BORDER_CONSTANT,value=0)
    else: # height > width
        image = cv2.copyMakeBorder(asarray(image), 0, 0, math.ceil((height-width)/2), math.floor((height-width)/2), 0, 0, cv2.BORDER_CONSTANT,value=0)
    image = cv2.resize(image, (512, 512)) # CAN be minor rescaling, fixes fluxuations of 1 in either direction due to rounding
    im = Image.fromarray(image)
    im.save(f"data/ISIC/{partition}/labels/ISIC_{mask}_segmentation.png")


def mask_to_txt(mask, partition):
    """
    Extracts bounding box information from mask, creates txt file
    """
    image = Image.open(f'data/ISIC/{partition}/masks/ISIC_{mask}_segmentation.png')
    width, height = image.size
    m = 512/width
    image = cv2.resize(asarray(image), None, fx=1, fy=1)
    ys, xs = np.where(image == 255) # extract x and y coordinates of white pixels (=255)
    x_center = int(np.median(xs))
    width = np.max(xs) - np.min(xs) 
    y_center = int(np.median(ys))
    height = np.max(ys) - np.min(ys)
    with open(f"data/ISIC/{partition}/labels/ISIC_{mask}.txt", "w") as file:
        file.write(f"0 {x_center/512} {y_center/512} {width/512} {height/512}") # write txt file


def scan_directory(partition):
    imgs = []
    for filename in os.scandir(f"data/ORIGINAL_ISIC/{partition}/labels"):
        if filename.is_file():
            imgs.append(str(filename.path[32+len(partition):-17]))
    if "" in imgs:
        imgs.remove("")
    return imgs

def extract(partition):
    for file in scan_directory(partition):
        # resize_img(file, partition)
        mask_to_txt(file, partition)

extract("val")

def test_extraction(mask):
    image = Image.open(f'data/ISIC/train/imgs/ISIC_{mask}.jpg')
    f = open(f"data/ISIC/train/txt/ISIC_{mask}.txt", "r")
    obj, x_center, y_center, width, height = np.array(f.read().split(" "))
    x_center, y_center, width, height = float(x_center)*512, float(y_center)*512, float(width)*512, float(height)*512
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle(((x_center) - (width)/2, (y_center) - (height)/2), (width), (height), linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


# test_extraction("0000001")

# GPU
# python train.py --workers 8 --device 0 --batch-size 32 --data data/ISIC/isic.yaml --img 512 512 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
# python train.py --batch 16 --epochs 2 --data data/ISIC/isic.yaml --img 512 512 --weights 'yolov7.pt'


# LOCAL
# python train.py --batch 32 --epochs 200 --data data/ISIC/isic.yaml --img 512 512 --weights 'yolov7.pt'

# python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 512 --source data/resized_valid_imgs/ISIC_0001769.jpg