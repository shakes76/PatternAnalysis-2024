
---
*Praneet Dhoolia (47364417)*
# Classifying Alzheimer’s Disease using a Vision Transformer (ViT)
This project aims to classify Alzheimer’s disease (normal and AD) using ADNI brain data and a Vision Transformer model, specifically `vit_small_patch16_224`. The primary objective is to achieve a test accuracy above 80% on the ADNI test set.

## Model
### ViT Architecture:
A Vision Transformer (ViT) is a deep learning model that applies the transformer architecture, traditionally used in natural language processing, to image recognition tasks. Unlike conventional convolutional neural networks (CNNs), which analyze image data locally through sliding filters, the Vision Transformer uses self-attention to capture global relationships among image patches.

<p align="center">
    <img width="700" src="assets/transformer.png">
</p>

**1. Patch Embedding Creation:** The input image is split into fixed-size patches, each of which is flattened and transformed into a vector through a linear embedding layer. This process converts each patch into a sequence of embeddings, allowing the image to be represented as a series of tokens (similar to words in text), enabling processing by the transformer.

**2. Positional Encoding for Spatial Awareness:** Since transformers lack inherent spatial understanding, positional encodings are added to each patch embedding. These encodings provide information about each patch’s position in the original image, ensuring the model can interpret spatial relationships effectively across patches.

**3. Class Token for Image-Level Prediction:** A learnable class token is prepended to the sequence of patch embeddings, representing the overall class of the image. As it passes through the model, this token aggregates information from all patches, enabling it to capture a global representation of the entire image by the model’s end.

**4. Self-Attention Mechanism in the Transformer Encoder:** The patch embeddings, along with the class token, pass through multiple layers of transformer encoders. The self-attention mechanism enables the model to capture both local and global dependencies across patches. Unlike CNNs, which primarily capture local features, self-attention allows each patch to relate to every other patch, providing a holistic view of the image.

**5. Classification Layer:** After passing through the encoder layers, the class token contains a learned representation of the entire image. This token is then fed into a final linear layer for classification, predicting the image’s class based on the relationships it has learned across patches.

## Training and Results
### Training Process
The training dataset contains images labeled as AD (Alzheimer’s Disease) and NC (Normal Control), organized into training, validation, and test sets.

**Preprocessing**:
- **Dataset Split**: The 20,000 images are split 70/30, providing 6,000 images in the validation set (3,000 AD and 3,000 NC).
- **Grayscale Conversion**: Each image is converted to grayscale to reduce computation by using a single channel.
- **Data Augmentation**: Random vertical and horizontal flips and rotations (up to 30 degrees) are applied to increase generalization.

**Hyperparameters**:
The following hyperparameters are used in the training process:

| Hyperparameter        | Value            |
|-----------------------|------------------|
| **Optimizer**         | Adam             |
| **Learning Rate**     | 0.0001           |
| **Scheduler**         | StepLR           |
| **Scheduler Step Size** | 10 epochs      |
| **Scheduler Gamma**   | 0.1              |
| **Epochs**            | 13               |
| **Batch Size**        | 32               |
| **Mixed Precision**   | Enabled (torch.amp) |
| **Criterion**         | CrossEntropyLoss |

### Training, Validation, and Testing Results
Below are the plots and statistics from the training process:

#### Epoch Output
<p align="center">
    <img width="700" src="assets/output.png">
</p>

#### Loss vs Epochs
<p align="center">
    <img width="500" src="assets/loss_vs_epochs.png">
</p>

#### Accuracy vs Epochs
<p align="center">
    <img width="500" src="assets/accuracy_vs_epochs.png">
</p>

#### Confusion Matrix
<p align="center">
    <img width="300" src="assets/confusion_matrix.png">
</p>

### Observations and Future Improvements
- The training process showed consistent progress in reducing training loss; however, a significant gap between training and validation accuracies indicates overfitting.
- **Potential Model Adjustments**:
  - Experiment with alternative Vision Transformer variants.
  - Increase dataset size or try more diverse augmentations.
  - Further tune hyperparameters to better balance training and validation accuracy.

### Usage
To install required libraries:
```bash
cd recognition/47364417
pip install -r requirements.txt
```

To train the model:
```bash
cd recognition/47364417
python train.py
```

To predict accuracy on the test set using a trained model:
```bash
cd recognition/47364417
python predict.py
```
