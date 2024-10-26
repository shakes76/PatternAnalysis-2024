# Medical Image Segmentation Project  

## Project Overview  
This project aims to utilize a 3D U-Net network for the segmentation of medical images. By loading MRI medical images and their corresponding labels, the model is trained and evaluated on a test set.  

## File Structure  
Below are the main files in the project and their functionalities:  

1. **train.py**:   
   - This script is used for training the model. It loads the training dataset and labels, trains the 3D U-Net model using an optimizer and loss function, and performs validation after each training epoch.  
   - It uses a custom dataset `MedicalDataset3D` and implements Dice loss as the training objective.  

2. **dataset.py**:   
   - Contains the custom dataset class `MedicalDataset3D`, which is used to load 3D medical images and optional labels.  
   - Provides functionality for loading image paths and supports data augmentation.  

3. **utils.py**:  
   - Contains the `DiceLoss` class for calculating the Dice loss and functions for computing the Dice coefficient.  
   - This module is primarily used for calculating the model's accuracy during the evaluation process.  

4. **modules.py**:  
   - Defines the structure of the 3D U-Net model, including convolutional layers, activation layers, and upsampling layers.  
   - The model is designed to extract features from the input images to improve segmentation accuracy.  

5. **predict.py**:  
   - This script is used to load test data and apply the trained model for predictions. It calculates and prints the average Dice coefficient on the test data while visualizing the input images, target labels, and prediction results.  

## Usage Instructions  

### Environment Requirements  
- Python 3.x  
- PyTorch  
- nibabel  
- numpy  
- tqdm  
- matplotlib  

### Data Preparation  
Before using this project, please ensure you have the following data prepared:  
- 3D medical images (NIfTI format)  
- Corresponding label data  

### Training the Model  
1. Modify the data paths in `train.py` to point to your dataset.  
2. Run `train.py` to start training.  

### Prediction  
1. Set the path for the test data in `predict.py`.  
2. Run `predict.py` to obtain prediction results and Dice coefficient evaluation.  

## Contribution  
Contributions of any kind are welcome! If you have suggestions or issues, please open an issue or pull request.  

## License  
This project is licensed under the MIT License. See the LICENSE file for details.