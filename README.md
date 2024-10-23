## Prostate 3D Segmentation with 3D U-Net

## Description
This project implements a 3D U-Net for segmenting 3D medical images of prostate data. 
The algorithm trains and tests a model on volumetric MRI data. 
It attempts to accurately predict the segmentation masks for multiple labels. 
The goal is to achieve a minimum Dice similarity coefficient of 0.7 for all labels 
in the test set, ensuring high segmentation accuracy.

## How It Works
The 3D U-Net architecture takes 4D volumetric MRI data as input 
and outputs segmentation maps for multiple classes. 
The model is trained using Dice Loss to maximize the overlap between predicted segmentation 
and ground truth. Key components include downsampling, bottleneck, and upsampling paths, 
which help capture spatial features at different scales. The network learns to extract 
and refine these features through convolutional layers and skip connections.

## Dependencies
- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- tqdm >= 4.62.0
- nibabel >= 3.2.1
- numpy >= 1.19.5

## Dataset Download Instructions

To download the Prostate 3D dataset required for this project, please follow these steps:

1. **Download the Dataset**: Visit the following link to access the dataset:
   [Prostate 3D Dataset](https://data.csiro.au/collection/csiro:51392v2?redirected=true)
   
   Click on the "Download" button to download the dataset files. The dataset is usually provided in a compressed format.

2. **Extract the Dataset**: Once the download is complete, extract the contents of the compressed file. Ensure that the folder structure is maintained.

3. **Place the Dataset in the Current Directory**: Move or copy the extracted dataset files into the current directory where your project is located. This will allow the program to import the data correctly.

The program expects the dataset to be in the current directory to function properly. 
Make sure that the dataset files are accessible to avoid any import errors during training or testing.

## Example Usage
python test_driver.py

## Preprocessing
This project preprocesses 3D medical images for segmentation using custom functions. 
The load_data_3D() function loads NIfTI files, and converts labels to one-hot encoding. 
The CustomDataset class efficiently loads image-label pairs and applies transformations, 
returning data in the expected [channels, depth, height, width] format for the model. 
Transformations include resizing and normalization.

## Training, Validation, and Testing Splits
The dataset is divided into training, and test sets with a split ratio 
of 70%/30%. The training set is used for optimizing model parameters and 
the test set for evaluating model performance.

## File Structure
/unet3d_shannon_searle
  ├── dataset.py                 # Manages dataset loading and transformations
  ├── model.py                   # Implements the 3D U-Net model architecture
  ├── train.py                   # Script for training the 3D U-Net model
  ├── predict.py                 # Evaluates the accuracy of the trained model
  ├── test_driver.py             # Executes the full model pipeline
  ├── /Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/
      ├── /data/
         ├── /semantic_labels_anon/    # Contains anonymized semantic labels
         ├── /semantic_MRs_anon/       # Contains anonymized semantic MR images

## Results
The model successfully segments the prostate and achieves a minimum Dice 
similarity coefficient of 0.7 on all test labels. 

## References
CSIRO Data Collection (2023). "Prostate-MRI-Histopathology". CSIRO Data Access Portal. https://data.csiro.au/collection/csiro:51392v2
OpenAI's ChatGPT (2024). "Code support and writing guidance provided by ChatGPT."