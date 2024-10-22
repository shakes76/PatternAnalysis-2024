# Alzheimer's Disease Classification using SWIN Transformer

# Overview

Alzheimer disease (AD) is a neurodegenerative disorder characterized by β-amyloid (Aβ)­
containing extracellular plaques and tau-containing intracellular neurofibrillary tangles,
The presentation of AD withshort-term memory difficulty is most common but impairment in expressive speech,
visuospatial processing and executive (mental agility) functions also occurs (ADD CITE).

For obvious reasons the early detection of AD is crucial for the treatment and management of the disease.
In this project, we aim to classify MRI images of the brain as either AD or non-AD using the Swin Transformer architecture.


The Swin Transformer is a hierarchical vision transformer that uses shifted windows to capture long-range dependencies in images. The Swin Transformer architecture is shown below.

![Swin Transformer Architecture](images/image.png)

The Swin Transformer architecture consists of a series of stages, each of which contains a series of Swin blocks. Each Swin block consists of a window-based self-attention mechanism followed by a feedforward neural network. The Swin blocks are connected in a hierarchical manner, with the output of one block being passed to the next block in the hierarchy. The Swin Transformer architecture is designed to capture long-range dependencies in images by using shifted windows to process the image in a hierarchical manner.    (ADD CITE)

# Data Description

The dataset used in this project is the ADNI dataset, which contains MRI images of the
brain from patients with Alzheimer's disease (AD) and healthy controls. The dataset
contains a total of 30000 images, split into 21000 training images and 9000 test images.
The test images were further split into 4500 validation images and 4500 test images.
There is approximately an equal number of AD and non-AD images in the dataset.

# Transformations

The images in the dataset were preprocessed using the following transformations:

1. Resize: The images were resized to 224x224 pixels to match the input size of the Swin Transformer model.
2. Normalize: The pixel values of the images were normalized to have a mean of 0.5 and a standard deviation of 0.5.
3. Random Horizontal Flip: The images were randomly flipped horizontally with a probability of 0.5.
4. Random Rotation: The images were randomly rotated by an angle between -10 and 10 degrees.
5. Random Crop: The images were randomly cropped to 224x224 pixels.

# Usage

## 1. Clone the repository

Clone the repository and navigate to the
alzheimer-classification directory.

```bash
$ git clone
$ cd alzheimer-classification
```

## 2. Install dependencies

Install the required dependencies using pip.

```bash
$ pip install -r requirements.txt
```

## 3. Download the dataset
```bash
$ python download.py
```

## 4. Adjust data directory
Run preprocess.py to adjust file structure to be suitable for train.py

```bash
$ python preprocess.py
```

Or manually adjust the file structure to be as follows:
```
wills_harrison_s4698315_adni_transformer/
    train/
        AD/
            ad_001.jpg
            ad_002.jpg
            ...
        ND/
            nd_001.jpg
            nd_002.jpg
            ...
    test/
        AD/
            ad_001.jpg
            ad_002.jpg
            ...
        ND/
            nd_001.jpg
            nd_002.jpg
            ...
```

## 5. Adjust hyperparameters and train the model

Adjust config.yaml to set the hyperparameters for the model. To train the model, run the following command.

```bash
$ python train.py
```

## 6. Evaluations
To evaluate the accuracy of the model, run the following command.

```bash
$ python evaluate.py --accuracy
```

To produce the confusion matrix, run the following command.

```bash
$ python confusion_matrix.py --confusion_matrix
```

To predict the class of an image, run the following command.

```bash
$ python predict.py --image_path <path_to_image>
```



Knopman, D. S., Amieva, H., Petersen, R. C., Chételat, G., Holtzman, D. M., Hyman, B. T., Nixon, R. A., & Jones, D. T. (2021). Alzheimer disease. Nature Reviews. Disease Primers, 7(1), 33–33. https://doi.org/10.1038/s41572-021-00269-y

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. arXiv. https://arxiv.org/abs/2103.14030
