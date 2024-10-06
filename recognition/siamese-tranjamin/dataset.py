import pydicom
import os
import numpy as np
from enum import Enum
import tensorflow as tf
from typing import Optional
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

class ImageUniformityOptions(Enum):
    CROP = 0
    RANDOMCROP = 1
    RESIZE = 2

class DicomDataset():
    def __init__(self, 
                 folder: str, 
                 resize_option: ImageUniformityOptions, 
                 limit: Optional[int] =None, 
                 resize_size: Optional[tuple[int, int]] = None,
                 normalise: bool = True,
                 **kwargs):
        dicom_data = []
        dicom_images = []
        dicom_labels = []

        labels = pd.read_csv(os.path.join(folder, "../train_labels.csv"))
        

        for ind, filename in enumerate(os.listdir(folder)):
            if ind == limit:
                break
            dicom = pydicom.dcmread(os.path.join(folder, filename))
            dicom_data.append(dicom)

            image = dicom.pixel_array
            label = labels.loc[labels["image_name"] == filename[:-4]].iloc[0]["target"]

            if normalise:
                image = image/255.

            match resize_option:
                case ImageUniformityOptions.RESIZE:
                    image = tf.image.resize(image, resize_size)

            dicom_images.append(image)
            dicom_labels.append(label)

        self.dicom_data = dicom_data
        self.dicom_labels = np.array(dicom_labels)
        self.dicom_images = np.array(dicom_images)
    
    def __len__(self):
        return self.dicom_images.shape[0]
    
    def show_sample_images(self, n_images):
        size = math.ceil(math.sqrt(n_images))
        fig, axs = plt.subplots(size, size)
        for i in range(n_images):
            plt.subplot(size, size, i + 1)
            index = random.randint(0, len(self) - 1)
            plt.imshow(self.dicom_images[index])
        plt.show()


dataset = DicomDataset("datasets/train", ImageUniformityOptions.RESIZE, limit=1000, resize_size=(2000, 2000))



pass