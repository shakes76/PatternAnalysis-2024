from dataset import load_data_2D
# this file is just for progressive testing of incomplete functionality

imageNames = [f"C:/Users/rjmah/Documents/Sem2 2024/COMP3710/HipMRI_study_keras_slices_data/keras_slices_train/case_004_week_0_slice_{i}.nii/case_004_week_0_slice_{i}.nii" for i in range(4)]
print(imageNames)
imgArrs = load_data_2D(imageNames)
print(imgArrs.shape)
