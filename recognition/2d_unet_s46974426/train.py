import os
import numpy as np
from dataset import load_data_2D

# folder that contains slices
image_folder = r'C:/Users/rober/Desktop/COMP3710/keras_slices_seg_train'
image_filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.nii.gz')]

# test loading the images
images = load_data_2D(image_filenames, normImage=True, categorical=False, dtype=np.float32, early_stop=False)
images, affines = load_data_2D(image_filenames, normImage=True, getAffines=True)
print(f'Loaded {len(affines)} affines')

import matplotlib.pyplot as plt

# display one imaage to confirm loading was correct
plt.imshow(images[0, :, :], cmap='gray')
plt.title('First Image Slice')
plt.axis('off')
plt.show()