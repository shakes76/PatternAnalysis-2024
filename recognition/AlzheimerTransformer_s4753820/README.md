# Alzheimer's Disease Detection using Vision Transformer (ViT)
Yuvraj Fowdar, 47538209.

## Project Overview

This project focuses on using Vision Transformers (ViT) for the detection of Alzheimer's Disease from MRI brain scans, leveraging the ADNI dataset. The goal is to classify brain images into two categories:

- Alzheimer's Disease (AD)
- Healthy (No Alzheimer's)

The Vision Transformer model is utilized to process brain images and make predictions with a target test accuracy of 80%. This project includes dataset preprocessing, augmentation, and the implementation of a Vision Transformer architecture for image classification.


## Usage 
### Repository Structure

- `predict.py`: Contains the code for making predictions using the trained ViT model. Softmax is applied externally for generating probability distributions for predictions. Can either generate classification report, or a  
- `dataset.py`: Handles dataset loading and augmentation techniques. This includes resizing the images to 224x224 and applying normalization based on empirical statistics or ImageNet statistics.
- `modules.py`: Includes the implementation of the Vision Transformer model and the necessary transformer blocks.
- `train.py`: The main training script to run experiments with different configurations such as number of transformer layers and batch size.

```
/
├── dataset.py
├── modules.py
├── train.py
├── predict.py
├── logs/
├── models/
├── plots/
```

### Install
**Set up your environment**:
- Create a virtual environment using venv or conda.
- Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
### Training

Run the following command to train the model, adjusting the hyperparameters and dataset path as needed:

```bash
python train.py --transformer_layers 8 --num_epochs 50 --embedding_dims 256 --mlp_size 1024 --num_heads 8 --exp_name "your_experiment_name" --path "/path/to/dataset"
```
Or a simpler command to use transformer defaults:
```bash
python train.py --exp_name "your_experiment_name" --path "/path/to/dataset"
```

After training, the model, logs, and plots will be saved in the respective directories (`models/`, `logs/`, `plots/`), with names with respect to `--exp_name`.

### Testing
On predict.py, use the following terminal command to test the model and perform one of two actions:

- `--run predict`: Get the test loss/accuracy of the trained model. (Default if --run is not specified).
- `--run brain-viz`: Visualize model predictions and display image grids with predicted labels and probabilities.

Examples:

Assuming there were no changes in model architecture (i.e defaults were used for train.py), then the below command will work.
```bash
python predict.py --model_path models/best_model.pth
```

If the trained model used different architectures (via --arg lines), the same changes must be put for the prediction.

```bash
python predict.py --model_path models/v10best_model.pth --run predict --num_transformer_layers 16
```
This command loads a model saved as v10best_model.pth (which was trained with 16 transformer layers, and the rest as transformer defaults) and outputs the test loss and accuracy as well as a classification report saved under `/plots`.

For image visualisation:
```bash
python predict.py --model_path models/v10best_model.pth --run brain-viz --num_transformer_layers 16
```
This visualizes the predictions for a batch of test images, displaying both true labels and predicted probabilities (using softmax). 



## Vision Transformer Architecture
{INFO ABOUT VIT; CITE THE REFERENCES HERE. ADD IMAGES. YAY.}


## Preprocessing: Dataset Loading + Augmentation
To ensure optimal performance and prevent data leakage, the dataset is split into training, validation, and test sets. The following preprocessing steps are applied:

1. Dataset Organization
The dataset, sourced from the ADNI database, is structured into two primary categories—Alzheimer’s Disease (AD) and No Alzheimer’s (NC)—for both training and test sets. The directory structure is as follows:
```
ADNI/
├── train/
│   ├── AD/
│   └── NC/
└── test/
    ├── AD/
    └── NC/
```
Each subdirectory (AD and NC) contains brain MRI images that are categorized accordingly.

2. Image Size and Normalization
After experimenting with different data augmentations (no normalisation, only normalisation, and aggressive augmenting + normalising), we found aggressive augmenting to work best.

The original vision transformer uses 16x16 image patches, and as our images are by default 256x256, we resize them to 224x224 pixels. Additionally, normalization is applied based on statistics computed from the training dataset to stabilize the model during training. These statistics include the mean (0.1156) and standard deviation (0.2229) for each RGB channel (in this case, all channels have equivalent statistics as they are greyscale images). Normalization ensures that the pixel values fall within a similar range, which helps the model converge faster.

Mean and standard deviation are computed only on the training dataset to avoid **data leakage** and are then applied consistently across training, validation, and test sets.

3. Data Augmentation
Data augmentation techniques are applied to the training dataset to improve the model’s robustness and generalization. These include:

Random Horizontal Flip: Randomly flips the image horizontally with a 50% probability.
Random Rotation: Randomly rotates the image by small angles (up to ±10 degrees).
Random Resized Crop: Randomly crops the image and resizes it to the target dimensions (224x224), using a scale of 0.8 to 1.0 of the original image size.
Center Crop: A smaller center crop (224 // 1.2) is applied after resizing.

## Experiments


## References

Ballal, A., 2023. Building a Vision Transformer from Scratch in PyTorch, Available at: https://www.akshaymakes.com/blogs/vision-transformer [Accessed 21 October 2024].

Dosovitskiy, A., et. al., 2020. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. Available at: https://arxiv.org/pdf/2010.11929.pdf [Accessed 21 October 2024].

Learnpytorch.io, 2023. PyTorch Paper Replicating. Available at: https://www.learnpytorch.io/08_pytorch_paper_replicating/#8-putting-it-all-together-to-create-vit [Accessed 21 October 2024].


Lightning.ai, 2023. Vision Transformer (ViT) - PyTorch Lightning Tutorial. Available at: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html [Accessed 21 October 2024].
