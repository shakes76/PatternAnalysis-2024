"""
Brief:
Segment the (downsampled) Prostate 3D data set (see Appendix for link) with the 3D Improved UNet3D [3] 
with all labels having a minimum Dice similarity coefficient of 0.7 on the test set. See also CAN3D [4] 
for more details and use the data augmentation library here for TF or use the appropriate transforms in PyTorch. 
You may begin with the original 3D UNet [5]. You will need to load Nifti file format and sample code is provided in Appendix B. 
[Normal Difficulty- 3D UNet] [Hard Difficulty- 3D Improved UNet]

"""