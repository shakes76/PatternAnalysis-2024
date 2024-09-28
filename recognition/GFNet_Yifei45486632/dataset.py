import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the images and basic data preprocessing
def load_images(directory):
    images = []
    labels = []

    for label in os.listdir(directory):
        label_folder =os.path.join(directory, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                file_path =os.path.join(label_folder,filename)
                try:
                    img = load_img(file_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return images,labels

# Test the result
if __name__ == "__main__":
    train_images, train_labels = load_images('train')
    test_images,test_labels = load_images('test')

    print(f"Loaded {len(train_images)} training images.")
    print(f"Shape of the image:{train_images[0].shape}")
    print(f"Total number of training labels: {len(train_labels)}")

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i +1)
        plt.imshow(np.uint8(train_images[i]))
        plt.title(train_labels[i])
        plt.axis('off')
    plt.show()