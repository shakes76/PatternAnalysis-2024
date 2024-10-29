# ADNI Brain Classification with Vision Transformer

This project implements a Vision Transformer (ViT) based classification system for analyzing brain images from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. The model classifies brain images into different categories: Cognitive Normal (CN), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD), and Subjective Memory Complaints (SMC).

## Features

- Custom-designed Vision Transformer for grayscale medical images
- Enhanced model architecture with:
  - Data normalization
  - Dropout regularization
  - Feature augmentation
  - Residual connections
  - Label smoothing support
  - Proper image size handling
- Cross-validation support with model ensemble capabilities
- Comprehensive evaluation metrics and visualization
- Dataset handling with proper train/validation/test splits

## Project Structure

```
.
├── dataset.py        # Data loading and preprocessing
├── modules.py        # Model architecture definitions
├── predict.py        # Evaluation and prediction scripts
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

# Initialize basic model
model = ViTClassifier(num_classes=4)

# Initialize enhanced model
enhanced_model = EnhancedViTClassifier(
    num_classes=4,
    dropout_rate=0.2,
    feature_dropout=0.1,
    image_size=224
)
```

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
- `confusion_matrix.png`: Visualization of model predictions
- `roc_curves.png`: ROC curves for each class
- `evaluation_metrics.json`: Detailed performance metrics

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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[Add your license information here]

## Acknowledgments

- ADNI for providing the dataset
- Vision Transformer (ViT) original implementation
- [Add any other acknowledgments]



