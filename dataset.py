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
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)


# for filename in os.listdir(ground_truth_folder):
#     if filename.endswith('.png'):
        
#         img = cv2.imread(os.path.join(ground_truth_folder, filename), cv2.IMREAD_GRAYSCALE)

#         contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         label_file = os.path.join(output_folder, filename.replace('.png', '.txt'))

#         with open(label_file, 'w') as f:
#             for contour in contours:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 # Normalize values to [0, 1]
#                 img_width, img_height = img.shape[1], img.shape[0]
#                 center_x = (x + w / 2) / img_width
#                 center_y = (y + h / 2) / img_height
#                 norm_width = w / img_width
#                 norm_height = h / img_height
#                 f.write(f'0 {center_x} {center_y} {norm_width} {norm_height}\n')

class DataSetProcessorTrainingVal():
    def __init__(self, training_data_path, annotated_data_path, transform=None):
        self.training_data = training_data_path
        self.annotated_data = annotated_data_path
        self.images = [
            img for img in os.listdir(training_data_path) 
            if img not in ["ATTRIBUTION.txt", "LICENSE.txt"]
        ]
        self.transform = transform
        print(len(self.images))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.training_data, self.images[index])

        if image_name in [
            "ATTRIBUTION.txt",
            "LICENSE.txt"
        ]:
            return None, None


        image = Image.open(image_path).convert("RGB")
        print(image_path)
        # print(self.annotated_data)
        # print(self.images[index])
        label_path = os.path.join(self.annotated_data, self.images[index].replace('.jpg', '_segmentation.txt'))



        with open(label_path, 'r') as file:
            labels = file.readline().strip()
            labels = [float(value) for value in labels.split()]


        image = self.transform(image)
        
        image_id = index  # or you can derive from file name if needed
        labels.insert(0, image_id % 16)

        labels = torch.tensor(labels, dtype=torch.float32, requires_grad=True)
        
        return image, labels