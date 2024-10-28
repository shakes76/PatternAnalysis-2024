from dataset import load_data_2D
#import tensorflow as tf
import matplotlib.pyplot as plt
# this file is just for progressive testing of incomplete functionality

imageNames = [f"C:/Users/rjmah/Documents/Sem2 2024/COMP3710/HipMRI_study_keras_slices_data/keras_slices_train/case_004_week_0_slice_{i}.nii/case_004_week_0_slice_{i}.nii" for i in range(4)]
labelNames = [f"C:/Users/rjmah/Documents/Sem2 2024/COMP3710/HipMRI_study_keras_slices_data/keras_slices_seg_train/seg_004_week_0_slice_{i}.nii/seg_004_week_0_slice_{i}.nii" for i in range(4)]
print(imageNames)
imgArrs = load_data_2D(imageNames, normImage=True)
labelArrs = load_data_2D(labelNames, categorical=True)
print(imgArrs.shape)
print(labelArrs.shape)

# visualise one of the images and its labels
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(imgArrs[0], cmap='gray')
ax1.set_title('Image')
ax1.axis('off')

ax2.imshow(labelArrs[0], cmap='viridis')
ax2.set_title('Label')
ax2.axis('off')

plt.tight_layout()
plt.axis('off')
plt.title('Black and White Visualization of imgArrs')
plt.show()