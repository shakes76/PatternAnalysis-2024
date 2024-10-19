# GFNet-ADNI
This repository contains the code for training a custom-made **GFNet** [1] model used to identify Alzheimer's disease in 2D sliced MRI brain scans. The model was trained on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, which contains a number of sliced MRI brain scan images categorised into Normal Control (NC) and Alzheimer's Disease (AD) classes. The model architecture is based on GFNet, an adapted vision transformer with global filtering designed to efficiently extract spatial patterns from images for classification.

<h3>Example images from ADNI dataset</h3>
<div style="display: flex; justify-content: space-between;">
    <img src="resources/image1.jpeg" style="width: 32%; height: auto;" alt="Image 1">
    <img src="resources/image2.jpeg" style="width: 32%; height: auto;" alt="Image 2">
    <img src="resources/image3.jpeg" style="width: 32%; height: auto;" alt="Image 3">
</div>


## The GFNet Algorithm
GFNet is a powerful neural network architecture that replaces the self-attention layers in vision transformers with global filters in the frequency domain, allowing it to capture both local and global image features effectively and efficiently.

![alt text](resources/intro.gif)

A key component of GFNet is its use of the Fast Fourier Transform (FFT) to handle the global filtering. GFNet applies a 2D discrete Fourier transform (DFT) to the input spatial features, transforming them into the frequency domain. Once in this domain, learnable global filters are applied through element-wise multiplication, allowing the network to efficiently capture global spatial dependencies. Finally, the inverse 2D Fourier transform (IFFT) is used to return the processed features back to the spatial domain. By leveraging FFT, GFNet reduces the computational complexity to log-linear (O(L log L)) as opposed to the quadratic complexity (O(L²)) seen in traditional self-attention models. This makes GFNet highly efficient while still retaining the ability to model both long-term and short-term interactions in the data​.

## Custom model dependencies
- Python 3.11.5
- torch>=2.4.1
- torchvision>=0.19.1
- timm>=1.0.10
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

The following data augmentation techniques were applied to the training set only to enhance model generalisability:
- random resize and crop
- random sharpness adjustments
- random horizontal flipping


## To use this model:
1. Clone the repository:
```
git clone https://github.com/shakes76/PatternAnalysis-2024
cd PatternAnalysis-2024/recognition/**GFNet-48336813**
```

2. Download dataset:

Download the [pre-processed](https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24) ADNI dataset.

3. Ensure the following folder structure:
```
GFNet-48336813/
             ├──dataset.py
             ├──modules.py
             ├──train.py
             ├──predict.py
             ├──utils.py
             ├──README.md
             ├──resources/
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
This will train the custom **GFNet** model on the ADNI dataset and save the best model to outputs/checkpoint_best.pth. During training for each epoch a summary of model statistics is printed as a single line in outputs/log.txt. Training loss and accuracy plots are saved in the plots/ directory.

4. Predict Disease:

```
python predict.py
```
This will use the best model from the previous training iteration against the test dataset and print the accuracy of the model and produce and save a confusion matrix. Then a single prediction is made by the model on one image selected at random from the test dataset. This image is saved to outputs with its true class label and the model's predicted class label.


## Results
The model was run with several different configurations which were part of the GFNet suite of pre-made configurations.
These included:

| Configuration | embed_dim  | Depth  |
|---------------|------------|------- |
| GFNet-Ti      | 256        | 12     |
| GFNet-XS      | 384        | 12     |
| GFNet-S       | 384        | 19     |
| GFNet-B       | 512        | 19     |



### Accuracy and Loss Plots



### Confusion Matrix:


## Acknowledgements
[1] Y. Rao, W. Zhao, Z. Zhu, J. Lu, and J. Zhou, “Global Filter Networks for Image Classification,” Oct. 26, 2021, arXiv: arXiv:2107.00645. Accessed: Oct. 16, 2024. [Online]. Available: http://arxiv.org/abs/2107.00645


