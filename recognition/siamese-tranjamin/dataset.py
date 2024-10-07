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
                 stratify: bool = True,
                 **kwargs):
        dicom_images = []
        dicom_labels = []

        labels = pd.read_csv(os.path.join(folder, "../train_labels.csv"))
        positive_labels = labels.loc[labels["target"] == 1]
        negative_labels = labels.loc[labels["target"] == 0]

        def process_image(filename):
            dicom = pydicom.dcmread(os.path.join(folder, filename))
            image = dicom.pixel_array
            if normalise:
                image = image/255.
            
            match resize_option:
                case ImageUniformityOptions.RESIZE:
                    image = tf.image.resize(image, resize_size)

            return image

        if stratify:
            for ind, filename in enumerate(positive_labels["image_name"]):
                if limit is not None and math.floor(limit/2) == ind:
                    break
                image = process_image(filename + ".dcm")
                label = 1

                dicom_images.append(image)
                dicom_labels.append(label)
            for ind, filename in enumerate(negative_labels["image_name"]):
                if limit is not None and math.floor(limit/2) == ind:
                    break
                image = process_image(filename + ".dcm")
                label = 0

                dicom_images.append(image)
                dicom_labels.append(label)
        else:
            for ind, filename in labels["image_name"]:
                if ind == limit:
                    break
                image = process_image(filename + ".dcm")
                label = labels.loc[labels["image_name"] == filename].iloc[0]["target"]

                dicom_images.append(image)
                dicom_labels.append(label)

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