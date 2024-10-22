# Alzheimer's Disease Detection using Vision Transformer (ViT)
Yuvraj Fowdar, 47538209.

## Project Overview

This project focuses on using Vision Transformers (ViT) for the detection of Alzheimer's Disease from MRI brain scans, leveraging the ADNI dataset. The goal is to classify brain images into two categories:

- Alzheimer's Disease (AD)
- Healthy (No Alzheimer's)

The Vision Transformer model is utilized to process brain images and make predictions with a target test accuracy of 80%. This project includes dataset preprocessing, augmentation, and the implementation of a Vision Transformer architecture for image classification.


## Usage 
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
After training, the model, logs, and plots will be saved in the respective directories (`models/`, `logs/`, `plots/`).

### Testing
Use the following terminal command to test the model and perform one of two actions:

- `--run predict`: Get the test loss/accuracy of the trained model.
- `--run brain-viz`: Visualize model predictions and display image grids with predicted labels and probabilities.

Examples:
```bash
python predict.py --model_path models/v10best_model.pth --run predict --num_transformer_layers 16
```
This command loads a model saved as v10best_model.pth (which was trained with 16 transformer layers) and outputs the test loss and accuracy.

For image visualisation:
```bash
python predict.py --model_path models/v10best_model.pth --run brain-viz --num_transformer_layers 16
```
This visualizes the predictions for a batch of test images, displaying both true labels and predicted probabilities (using softmax).


## Vision Transformer Architecture

## Repository Structure


- predict.py
Softmax applied onto model externally for probability predictions!!

## Preprocessing: Dataset Loading + Augmentation
- vision trasnfoemrs seem to want image sizes that are multiples of 16. So 224x224 might be the angle.
- Also need to normalise the data so we can help the model train better and converge faster.
- we tried multiple versiopns of data augmentation, settled down for this.
- 
Resize to 224x224.
<!-- Random Horizontal Flip.
Random Rotation (small angles, e.g., ±10 degrees).
Random Brightness/Contrast Adjustments (slight, e.g., ±20%). -->
Normalization using empirically gathered statistics from our ADNI dataset; basically found the normalisation from training dataset, applied this everywhere 
(only colelcted from train dataset so no data leakage occurs).

## Experiments


## References

Ballal, A., 2023. Building a Vision Transformer from Scratch in PyTorch, Available at: https://www.akshaymakes.com/blogs/vision-transformer [Accessed 21 October 2024].

Dosovitskiy, A., et. al., 2020. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. Available at: https://arxiv.org/pdf/2010.11929.pdf [Accessed 21 October 2024].

Learnpytorch.io, 2023. PyTorch Paper Replicating. Available at: https://www.learnpytorch.io/08_pytorch_paper_replicating/#8-putting-it-all-together-to-create-vit [Accessed 21 October 2024].


Lightning.ai, 2023. Vision Transformer (ViT) - PyTorch Lightning Tutorial. Available at: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html [Accessed 21 October 2024].
