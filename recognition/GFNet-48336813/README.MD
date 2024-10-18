# GFNet-ADNI
This repository contains the code for training a custom-made GFNet [1] model used to identify Alzheimer's disease in 2D sliced MRI brain scans. The model was trained on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, which contains a number of sliced MRI brain scan images categorised into Normal Control (NC) and Alzheimer's Disease (AD) classes. The model architecture is based on GFNet, a convolutional and global filtering-based architecture designed to efficiently extract spatial patterns from images for classification.

## The GFNet Algorithm
GFNet is a powerful neural network architecture that combines convolutional layers with global filters, allowing it to capture both local and global image features effectively. Unlike traditional CNNs that focus only on local features, GFNet can process the entire image context using global Fourier-like filtering mechanisms, which enhances its ability to recognise complex patterns across the whole image.

![alt text](images/intro.gif)

## Dependencies
- Python 3.11.5
- torch>=2.4.1
- torchvision>=0.19.1
- Pillow>=8.2.0
- scikit-learn>=0.24.2
- matplotlib>=3.4.2

### Development and testing environment
This project was developed on an Apple Macbook Pro with 32 GB of RAM and an M1 Max Chip. All device calls were implemented in a platform-agnostic manner, while CUDA-specific modifications from the original GFNet codebase were retained.

## Dataset
The dataset utilised in this project is a [pre-processed](https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24) version of the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The ADNI dataset contains MRI scans contributed longitudinally by over 2,500 participants from across the U.S. and Canada. These scans are classified into two main categories:

* Alzheimer's Disease (AD)
* Normal Control (NC)

The pre-processed ADNI dataset contained 30,520 images and was obtained with the following structure:
```
AD_NC/
   ├── train/        
   │   ├── AD/
   │   └── NC/
   ├── test/
       ├── AD/
       └── NC/
```
where the train and test split was approximately 70/30. The AD and NC classes within train and test were split roughly 50/50.

A validation dataset was also created by randomly removing 10% of the images from the
training set resulting in the following dataset structure:
```
data/
   ├── train/
   │   ├── AD/
   │   └── NC/
   ├── val/
   │   ├── AD/
   │   └── NC/
   └── test/
       ├── AD/
       └── NC/
```

### Data pipeline
To use the ADNI dataset a custom PyTorch dataset class was created for loading images and applying any necessary transformations such as resizing, normalisation, and data augmentation.

The follow preprocessing techniques were used for all datasets:
- image resizing to 224x224 (as was done in the GFNet paper)
- data normalisation (based off each dataset mean and standard deviation)

The following data augmentation techniques were applied to the training set only:
- Random resize and crop
- Random sharpness adjustments
- Random horizontal flipping


## To use this model:
1. Clone the repository:
```
git clone https://github.com/shakes76/PatternAnalysis-2024
cd PatternAnalysis-2024/recognition/GFNet-48336813
```

2. Download dataset:

Download the [pre-processed](https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24) ADNI dataset.

3. Ensure the following folder structure:
```
GFNet-48336813/
             ├──images/
             ├──outputs/
             ├──plots/
             ├──data/
                ├── test/
                │   ├── AD/
                │   └── NC/
                ├── train/
                │   ├── AD/
                │   └── NC/
                └── val/
                    ├── AD/
                    └── NC/
```

4. Train the Model:

```
python train.py
```
This will train the **GFNet** model on the ADNI dataset and save the best model to outputs/checkpoint_best.pth. During training for each epoch a summary of model statistics is printed as a single line in outputs/log.txt. Training loss and accuracy plots are saved in the plots/ directory.

4. Predict Disease:

```
python predict.py
```
This will predict the best model from the previous training iteration against the test dataset and print the accuracy of the model and produce and save a confusion matrix. Then the script will randomly select a single image from the test dataset and run inference using the model and save the image with its label and prediction.


## Results


### Accuracy and Loss Plots



### Confusion Matrix:


## Acknowledgements
[1] Y. Rao, W. Zhao, Z. Zhu, J. Lu, and J. Zhou, “Global Filter Networks for Image Classification,” Oct. 26, 2021, arXiv: arXiv:2107.00645. Accessed: Oct. 16, 2024. [Online]. Available: http://arxiv.org/abs/2107.00645


