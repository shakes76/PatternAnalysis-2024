import torch
import pandas as pd
import os
import cv2
from PIL import Image

# folder paths
train_dir = 'Data/ISIC2018_Task1-2_Training_Input_x2'
val_dir = 'Data/ISIC2018_Task1_Training_GroundTruth_x2'
test_dir = 'Data/ISIC2018_Task1-2_Test_Input'


# YOLO requires each groundtruth image to have a txt file with:
# class ID, Normalized center x, Normalized center y, Normalized width, Normalized height
ground_truth_folder = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/ISIC2018_Task1_Training_GroundTruth_x2'
output_folder = 'Data/train_labels'

# create labels
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for filename in os.listdir(ground_truth_folder):
    if filename.endswith('.png'):
        
        img = cv2.imread(os.path.join(ground_truth_folder, filename), cv2.IMREAD_GRAYSCALE)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_file = os.path.join(output_folder, filename.replace('.png', '.txt'))

        with open(label_file, 'w') as f:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Normalize values to [0, 1]
                img_width, img_height = img.shape[1], img.shape[0]
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                f.write(f'0 {center_x} {center_y} {norm_width} {norm_height}\n')
