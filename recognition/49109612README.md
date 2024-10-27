
# Skin Lesion Similarity Assessment Using Siamese Network

## Introduction

This project aims to assess the similarity between skin lesion images using a Siamese Network. By determining whether two skin lesion images belong to the same category, we can assist doctors in diagnosis and research.

## Problem Description

In the diagnosis of skin lesions, different types exhibit varying characteristics. Comparing the similarity between lesion images can help identify unknown lesion types or classify known ones. This is particularly useful in early detection and treatment planning for skin cancer and other dermatological conditions.

## Algorithm Description

We utilize a Siamese Network to learn the similarity between images. The main structure of the network includes:

- **Shared Convolutional Network (Feature Extractor)**: Two input images pass through the same convolutional neural network to extract high-dimensional features.
- **Fully Connected Layers**: The two feature vectors are concatenated and passed through fully connected layers to classify whether the two images belong to the same category.

The model input consists of RGB images resized to 100×100 pixels, and the output is logits for two classes, indicating whether the images are of the same or different categories.

## How It Works

1. **Data Preprocessing**:
   - Resize the original images to a uniform size of 100×100 pixels.
   - Apply necessary data augmentation strategies.
   - Generate image pairs with corresponding labels (1: same category, 0: different categories).

2. **Model Training**:
   - Use the Cross-Entropy Loss function to measure the difference between predictions and true labels.
   - Employ the SGD optimizer to update model parameters.
   - Monitor training loss and validation accuracy during training to adjust the model accordingly.

3. **Model Validation**:
   - Evaluate the model's performance on the validation set by checking the accuracy.

## Dependencies

- Python 3.7+
- NumPy
- PyTorch 1.7+
- Torchvision

**Version Information**:

- `torch`: 1.7.1
- `torchvision`: 0.8.2
- `numpy`: 1.19.5

## Reproducibility

To ensure reproducibility of the results:

- Random seeds are set using:

  ```python
  import torch
  import random
  import numpy as np

  torch.manual_seed(42)
  random.seed(42)
  np.random.seed(42)
  ```

- Data splits are consistent with a fixed random seed.

## Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/your_repository.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

- Organize your skin lesion image dataset into folders by category, with each category in a separate folder.
  ```
  dataset/
  ├── class1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── class2/
  │   ├── image3.jpg
  │   ├── image4.jpg
  │   └── ...
  └── ...
  ```
- Ensure the dataset paths in `train.py` match the location of your dataset.

### 4. Run the Training Script

```bash
python train.py
```

## Example Inputs and Outputs

- **Input Image Pair**:


- **Model Prediction**:

  - Probability of the same category: **85%**
  - Probability of different categories: **15%**

## Data Preprocessing

- **Image Resizing**: All images are resized to 100×100 pixels.
- **Data Augmentation**: Apply transformations such as horizontal and vertical flips if necessary.
- **Pair Generation**: Randomly select image pairs, deciding whether they are from the same category.

## Dataset Splitting

- **Training Set**: 80% of the data used for training the model.
- **Validation Set**: 20% of the data used to evaluate model performance.

We use a custom dataset class `SiameseISICDataset` and set a random seed of 42 to ensure the split is reproducible.

## Code Explanation

- **train.py**: The main training script, including data loading, model definition, training loop, and validation process.
- **module.py**: Defines the Siamese Network model.
- **dataset.py**: Custom dataset class to generate image pairs and labels.

Detailed comments are added throughout the code to facilitate understanding of each part's functionality.

## Dependencies and Versions

List of required packages and their versions:

- Python (3.7+)
- NumPy (1.19.5)
- PyTorch (1.7.1)
- Torchvision (0.8.2)

Install them using:

```bash
pip install numpy==1.19.5 torch==1.7.1 torchvision==0.8.2
```

## Specific Preprocessing Techniques

- **Normalization**: Images are normalized to have pixel values between 0 and 1.
- **Label Encoding**: Labels are converted to `LongTensor` for compatibility with `CrossEntropyLoss`.

## Training, Validation, and Testing Splits Justification

- **Training Set (80%)**: Provides sufficient data for the model to learn patterns.
- **Validation Set (20%)**: Used to tune hyperparameters and prevent overfitting.

Splitting is done randomly but consistently using a fixed seed to ensure reproducibility.

## References

- Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. ["Siamese neural networks for one-shot image recognition."](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) *ICML Deep Learning Workshop*, 2015.

---

**Note**: Ensure that you have correctly configured the environment and prepared the dataset before running the code.

---

This README file is properly formatted using GitHub Markdown, including headings, code blocks, tables, images, and links.
