# ADNI Brain Classification with Vision Transformer

This project implements a Vision Transformer (ViT) based classification system for analysing brain images from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. The model classifies brain images into different categories: Cognitive Normal (CN), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD), and Subjective Memory Complaints (SMC).

## Features

- Custom-designed Vision Transformer for grayscale medical images
- Enhanced model architecture with:
  - Data normalisation
  - Dropout regularisation
  - Feature augmentation
  - Residual connections
  - Label smoothing support
  - Proper image size handling
- Cross-validation support with model ensemble capabilities
- Comprehensive evaluation metrics and visualisation
- Dataset handling with proper train/validation/test splits

## Project Structure

```
.
├── dataset.py        # Data loading and preprocessing
├── modules.py        # Model architecture definitions
├── train.py         # Training and optimisation scripts
├── predict.py       # Evaluation and prediction scripts
└── AD_NC/           # ADNI dataset directory
    ├── train/
    │   ├── AD/
    │   └── NC/
    └── test/
        ├── AD/
        └── NC/
```

## Requirements

- PyTorch
- torchvision
- PIL (Python Imaging Library)
- numpy
- matplotlib
- seaborn
- scikit-learn
- pandas
- tqdm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install torch torchvision pillow numpy matplotlib seaborn scikit-learn pandas tqdm
```

3. Prepare the ADNI dataset in the following structure:
```
AD_NC/
├── train/
│   ├── AD/
│   └── NC/
└── test/
    ├── AD/
    └── NC/
```

## Usage

### Training Data Preparation

The dataset module (`dataset.py`) provides functionality for loading and preprocessing ADNI brain images. It includes:

```python
from dataset import get_dataset, get_dataloader

# Get training and validation datasets
train_dataset, val_dataset = get_dataset(train=True, val_proportion=0.2)

# Get test dataset
test_dataset = get_dataset(train=False)

# Alternatively, get dataloaders directly
train_loader, val_loader = get_dataloader(batch_size=64, train=True)
test_loader = get_dataloader(batch_size=64, train=False)
```

### Model Architecture

Two main model architectures are provided in `modules.py`:

1. `ViTClassifier`: Basic Vision Transformer adapted for grayscale images
2. `EnhancedViTClassifier`: Advanced version with additional features

```python
from modules import ViTClassifier, EnhancedViTClassifier

# Initialise basic model
model = ViTClassifier(num_classes=4)

# Initialise enhanced model
enhanced_model = EnhancedViTClassifier(
    num_classes=4,
    dropout_rate=0.2,
    feature_dropout=0.1,
    image_size=224
)
```

### Training the Model

The training process is handled by the `OptimizedTrainer` class in `train.py`. The system supports various training optimisations including:

- Mixed precision training
- Distributed training support
- Automatic batch size optimization
- Learning rate scheduling
- Early stopping
- Checkpoint management
- Comprehensive metrics tracking

To train the model:

```python
from train import train_model_optimized

# Initialise training with default parameters
model, history = train_model_optimized(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    epochs=20,
    batch_size=64,
    lr=1e-4,
    classes=['CN', 'MCI', 'AD', 'SMC'],
    early_stopping_patience=4
)
```

Alternatively, use the command line interface:

```bash
python train.py
```

The training script will:
1. Automatically detect and configure available hardware
2. Optimise training parameters based on system capabilities
3. Save checkpoints and training curves
4. Generate comprehensive training reports
5. Implement early stopping if validation metrics plateau

Training outputs include:
- Model checkpoints (regular intervals and best model)
- Training/validation curves
- Confusion matrix visualisations
- Complete training results in JSON format

### Evaluation and Prediction

The `predict.py` script provides comprehensive evaluation functionality:

```bash
python predict.py
```

This will:
1. Load trained models
2. Perform cross-validation
3. Generate evaluation metrics including:
   - Confusion matrix
   - ROC curves
   - Classification report
   - Per-class accuracy
4. Save results to a timestamped directory

### Output

The evaluation script generates the following outputs in a timestamped directory:
- `confusion_matrix.png`: Visualisation of model predictions
- `roc_curves.png`: ROC curves for each class
- `evaluation_metrics.json`: Detailed performance metrics
- `training_curves.png`: Training and validation metrics over time

## Model Performance

The system evaluates models using multiple metrics:
- Overall accuracy
- Per-class accuracy
- ROC curves with AUC scores
- Confusion matrix
- Detailed classification report including precision, recall, and F1-score

## Cross-Validation

The system supports model ensemble through cross-validation:
```python
predictions, true_labels, probabilities = cross_validate_models(
    model1_path="./checkpoints/model1.pt",
    model2_path="./checkpoints/model2.pt",
    test_loader=test_loader,
    device=device,
    classes=CLASSES
)
```

## License

Apache License - Version 2.0, January 2004 (http://www.apache.org/licenses/)

## Acknowledgments

- ADNI for providing the dataset
- Vision Transformer (ViT) original implementation
