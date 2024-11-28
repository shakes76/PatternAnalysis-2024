# Classification of Brain Images in the ADNI dataset using the GFNet Vision Transformer
### Benjamin Thatcher/s4784738


## Overview (Classifying Brain Images)
The ADNI dataset contains thousands of human brain images obtained via MRI scans. Some of them are from people who have Alzheimer's disease (AD) and some are from people who don't (normal control, or NC). This project aims to detect whether a patient has Alzheimer's disease or not using image classification techniques. Using the Global Filter Network, or GFNet, I trained a model on the brain images in the ADNI dataset that classifies brain images as Alzheimer's positive or negative with 73% accuracy.


## How a GFNet Works
The Global Filter Network is a neural architecture that replaces traditional convolutional layers with frequency-domain filters to capture global spatial relationships in images. Instead of relying on local receptive fields like in standard CNNs, the GFNet applies global filters using the fast Fourier transform, enabling it to model long-range dependencies more efficiently. This allows the GFNet to capture both local and global features, making it well suited for tasks where global structure is important, such as image classification.

![GFNet Architecture](./assets/GFnet_structure.gif)
_(Rao, 2024)_


## Dependencies
To run this code, the following dependencies must be installed:
-python 3.12.4
-torch 2.4.0
-torchvision 0.19.0
-numpy 1.26.3
-matplotlib 3.9.2
-timm 1.0.9
-pillow 10.2.0


## Installation
The dependencies above can be installed by running:
```
pip install torch torchvision numpy matplotlib timm pillow
```


## Directory Structure
```
recognition/
├── GFNet_s4784738/ 
│   ├── assets/
│   │   ├── GFNet_structure.gif
|   |   ├── training_accuracy.png
|   |   ├── training_loss.png
|   |   ├── validation_accuracy.png
|   |   └── validation_loss.png
│   ├── dataset.py
│   ├── modules.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   └── README.md 
└── README.md
```
Note that the /train/ and /test/ image folders (whose paths are specified in utils.py) should have /AD/ and /NC/ subfolders containing Alzheimers and normal control brain images, repectively.
```
├── train/
|   ├── AD/
|   └── NC/
└── test/
    ├── AD/
    └── NC/
```


## Data Pre-processing
The images in the ADNI dataset are pre-processed in dataset.py.
### Training and Validation Images
Images in the train directory of the AD_NC folder were used to construct the training and validation sets. The training-validation split I chose was 80-20.
These images were resized to 224x224 pixels, converted to greyscale, normalized, and converted to tensors.
To improve the model's ability to learn the brain image features, I tested transformations such as flipping and rotating the images at random, but the 
images already had a lot of this kind of variability in them, so these transformations had very little impact on the model's accuracy. I then tried some 
transformations that more directly impacted the image quality (ColourJitter, RandomAffine, RandomResizedCrop, and GaussianBlur) and found significantly more success. For this reason, these 4 transformations were applied to the training and validation images.

### Testing Images
Images in the test directory of the AD_NC folder were used to construct the testing split. This ensured that model could not learn the features of the testing data beforehand. 
The only pre-processing these images recieved was being resized to 224x224 pixels, converted to greyscale tensors, and normalization.


## Configuration settings
Before running the code, it's important to ensure that you have the desired configuration settings. The model configurations and hyperparameters are stored and retrieved from 
utils.py. They can also be found below:
```
epochs = 30
learning_rate = 1e-4
patch_size = (16, 16)
embed_dim = 512
depth = 19
mlp_ratio = 4
drop_rate = 0.1
drop_path_rate = 0.1
weight_decay = 1e-2
t_max = 6
```
The image paths are set up to run on UQ's rangpur cluster by default. If you are running the model anywhere else, please adjust the paths to the training and testing image directories as necessary in utils.py.
```
def get_path_to_images():
    train_path = '/home/groups/comp3710/ADNI/AD_NC/train'
    test_path = '/home/groups/comp3710/ADNI/AD_NC/test'

    return train_path, test_path
```
When running inference, you can set a specific image to be tested by setting the img_path variable in the get_prediction_image() function in utils.py.
```
def get_prediction_image():
    # A set image path to be returned
    img_path = _Your image path here_
```


## Running the Code
### Training the Model
The model can be trained by running
```
python train.py
```
This will also save both the model and plots of the training and validation, as well as run one epoch of inference on all images in the /test/ folder.
Model parameters and image sources can be changed as desired in utils.py.

### Predicting the Classification of an Image
You can run inference on a single image by running
```
python predict.py
```
By default, a random image will be chosen from the /test/ folder. You can set a specific image to be tested in utils.py.


## Plots
These plots were obtained by running training for 40 epochs with the default hyperparameters.
![Training Accuracy](./assets/training_accuracy.png)
Training Accuracy of the Model

![Training Loss](./assets/training_loss.png)
Training Loss of the Model

![Validation Accuracy](./assets/validation_accuracy.png)
Validation Accuracy of the Model

![Validation Loss](./assets/training_loss.png)
Validation Loss of the Model


## Results
When running inference on the test images, the model was able to predict the correct classification of the brain images with 73.0% accuracy.
When running predict.py with random images from the test set, similar behavior was observed. About 3/4 of the images were classified correctly, and most of the incorrectly classified images looked noticably different to other images of their kind when inspected.


## References
- Rao, Y., Zhao, W., Zhu, Z., Lu, J., and Zhou, J. (2021). Global filter networks for image classification. Advances in neural information processing systems.
- Original GFNet Code: [GFnet code on Github](https://github.com/raoyongming/GFNet)
