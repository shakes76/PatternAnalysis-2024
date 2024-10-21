# Classifying Alzheimer's Disease with a Vision Transformer



## The Global Filter Network (GFNet)
The Global Filter Network is a network that performs much of the training in the frequency domain through the use of the Fast Fourier Transform. This speeds up the training process signifcantly.


## Dataset
### Data Split
The ADNI dataset is already split into two sets for training and testing. Since a validation set was not provided, a portion of the training data was used for validation; specifically, 20% of the images in the provided train set was used.

The training data contained multiple  slices of the same brain of patients.
To ensure no data leakage between the training and validation sets, the data had to be split on a patient level. This was done by looking at the names of the files, extracting the `patient_id`, and assigning patients to either the training or validation set. This has all been implemented in the `dataset.py` file.

After the patient level split was complete,  the data was loaded into a `DataLoader` object. This increases the efficiency during training.


### Pre-processing
Pre-processing was also performed on the data before training. Specifically, the data was resized to 224 x 224, to align with the suggested sizes proposed by Rao., et al. [1]. The data was also normalised.

The provided ADNI dataset follows the directory structure below:
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

## Usage


## Dependencies
| Dependency | Version |
|------------|---------|
| torch      | >=1.8.0 |
| torchvision| latest  |
| timm       | latest  |

To quickly install all relevant dependencies, run the following command.
```
python -m pip install requirements.txt
```

### Notes
adjust learning rate
ensure no data leakage
data augmentation -- flipping y-axis
no cropping
add gaussian noise

in validation no spectral leakage
normalisation (either per image or average mean/std for whole dataset)