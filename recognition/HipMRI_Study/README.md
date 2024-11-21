# HipSRI Study on Prostate Cancer

### Objective
The objective of this study is to segment prostate cancer images from the HipSRI dataset using a **2D UNet model**, which is required for this task. The UNet model is well-suited for image segmentation tasks, particularly in medical imaging.

### Methodology
1. **Model Implementation**: The **UNet model** was implemented and saved in `modules.py` as a class. This class is used in `train.py` for training the model.
   
2. **Data Loading**: I downloaded the folder containing the 2D slice images, and used the provided Nifti file reader (Appendix B) to load the images into the Python environment. I made slight modifications to ensure uniform image sizing for training purposes.

3. **Model Training and Testing**: In `train.py`, the UNet model was trained using the preprocessed 2D slices. The `predict.py` script was used for inference and testing of the trained model on new data.

### Challenges Faced
- **Memory Issues**: The program required a substantial amount of memory during execution, leading to crashes on my local device.
  
- **Remote Execution Errors**: Attempts to run the program on the **Rangpur** cluster encountered issues with importing other Python files, resulting in `ModuleNotFoundError`. This made it difficult to complete model training and testing remotely.

### Conclusion
Although the code was written and the UNet model was properly implemented, memory constraints on my local device and remote system import errors on Rangpur have limited my ability to fully process and test the trained model. Further debugging or system configuration changes are needed to resolve these issues.
