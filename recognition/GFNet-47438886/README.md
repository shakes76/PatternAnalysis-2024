# Classifying Alzheimer's Disease with a Vision Transformer

This repository contains the code used to train one of the latest vision transformers, known as the GFNet, to identify Alzheimer's Disease in 2D slices of MRI brain scans. The data used comes from the [ADNI dataset](https://adni.loni.usc.edu/), which is split into two classes: AD (Alzheimer's Disease) and NC (Normal Control). The model architecture is based on the innovative work of Rong, et al [1].

## The Global Filter Network (GFNet)
The Global Filter Network is a network that performs much of the training in the frequency domain through the use of the Fast Fourier Transform. This speeds up the training process signifcantly.


## Dataset
The downloaded ADNI dataset follows the directory structure below:
```
│AD_NC/
├──test/
│  ├── AD
│  │   ├── 1003730_100.jpeg
│  │   ├── 1003730_101.jpeg
│  │   ├── ......
│  ├── NC
│  │   ├── 1182968_100.jpeg
│  │   ├── 1182968_101.jpeg
│  │   ├── ......
├──train/
│  │   ├── 218391_78.jpeg
│  │   ├── 218391_79.jpeg
│  │   ├── ......
│  ├── NC
│  │   ├── 1000359_100.jpeg
│  │   ├── 1000359_101.jpeg
│  │   ├── ......
```
The files are named as `patientID_sliceID.jpeg`, where `patientID` represents brains belonging to the same patient and `sliceID` representing the 2D slice number of a 3D MRI brain scan. 

### Data Split
The ADNI dataset is already split into two sets for training and testing. Since a validation set was not provided, a portion of the training data was used for validation; specifically, 20% of the images in the provided train set was used.

The training data contained multiple  slices of the same brain of patients.
To ensure no data leakage between the training and validation sets, the data had to be split on a patient level. This was done by looking at the names of the files, extracting the `patientID`, and assigning patients to either the training or validation set. This has all been implemented in the `dataset.py` file.

After the patient level split was complete,  the data was loaded into a `DataLoader` object. This allowed for the efficient batch loading and preprocessing of the data. 


### Pre-processing
Pre-processing was also performed on the data before training. Specifically, the data was resized to 224 x 224, to align with the suggested sizes proposed by Rao., et al. [1]. The data was also normalised.

## Dependencies
| Dependency | Version |
|------------|---------|
| torch      | latest  |
| torchvision| latest  |
| timm       | latest  |
| scikit_learn| latest |
| Pillow     | latest  |
| numpy      | latest  |
| matplotlib | latest  |

## Usage
1. Clone the repository:
```
git clone https://github.com/Kevin-Gu-2022-UQ/PatternAnalysis-2024.git
cd PatternAnalysis-2024/recognition/GFNet-4743888
```

2. Download the [dataset](https://adni.loni.usc.edu/), ensuring it matches the structure shown above.

3. Setup the environment. To quickly install all relevant dependencies, run the following command.
```
python -m pip install -r requirements.txt
```

4. Train the model by running the following command:
```
python train.py
```
This will train the model on 100 epochs. Every time the model improves in accuracy, it will be saved to `model_weights.pth`, which can be used with `predict.py` to obtain a classification accuracy on the test set. 

5. Run inference on test dataset:
```
python predict.py
```

## Inputs
#### Example Input for Brain with Alzheimer's Disease
![Example input of brain with Alzheimer's Disease](assets/218391_78_AD.jpeg)
#### Example Input for Brain without Alzheimer's Disease
![Example input of brain with no Alzheimer's Disease](assets/808819_88_NC.jpeg)




## Results
### Training and Validation Loss Plot

### Training Accuracy Plot

### Test Accuracy Plot

### Notes
adjust learning rate
ensure no data leakage
data augmentation -- flipping y-axis
no cropping
add gaussian noise

in validation no spectral leakage
normalisation (either per image or average mean/std for whole dataset)

Centre crop, add gaussian noise, 
cosine annealing with defaults
epochs 200
save state dict
vertical flipping
add saturation, etc. variability between MRI machines

confusion matrix: false positives, etc.

# References
[1] Y. Rao, W. Zhao, Z. Zhu, J. Lu, J, Zhou. (2021, October). "Global Filter Networks for Image Classification." [Online]. Available: https://arxiv.org/pdf/2107.00645

[2] Y. Rao, W. Zhao, Z. Zhu, J. Lu, J, Zhou. (2021, December. 7). *GFNet* [Online]. Available: https://github.com/raoyongming/GFNet