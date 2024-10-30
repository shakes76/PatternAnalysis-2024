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
├── README.md
├── dataset.py        # Data loading and preprocessing
├── modules.py        # Model architecture definitions
├── train.py         # Training and optimisation scripts
├── predict.py       # Evaluation and prediction scripts
├── AD_NC/           # ADNI dataset directory
│   ├── train/
│   │   ├── AD/
│   │   └── NC/
│   └── test/
│       ├── AD/
│       └── NC/
└── checkpoints      # Trained models
```

## Requirements
- python (3.11.2)
- PyTorch (2.0.0)
- torchvision (0.15.1)
- PIL (Python Imaging Library)
- numpy (1.26.4)
- matplotlib (3.7.1)
- seaborn (0.12.2)
- scikit-learn (1.2.1)
- pandas (1.5.3)
- tqdm (4.66.2)

## Installation

0. Install `python 3.11.2`

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install torch==2.0.0 \
            torchvision==0.15.1 \
            pillow \
            numpy==1.26.4 \
            matplotlib==3.7.1 \
            seaborn==0.12.2 \
            scikit-learn==1.2.1 \
            pandas==1.5.3 \
            tqdm==4.66.2
```

3. Prepare the ADNI dataset in the following structure:
Download from https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24
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
2. `EnhancedViTClassifier`: Advanced version with additional features, but the performance seems worse

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

#### Cross-Validation

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

**Classes & Overall Metrics**
| Metric | Value |
|--------|--------|
| Classes Present | CN, MCI |
| Test Set Size | 9000 |
| Overall Accuracy | 0.847 |
| Evaluation Date | 2024-10-30 07:05:15 |

**Per-Class Performance Metrics**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| CN | 0.786 | 0.957 | 0.863 | 4540 |
| MCI | 0.944 | 0.734 | 0.826 | 4460 |

**Aggregate Metrics**
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|-----------|----------|
| Macro Avg | 0.865 | 0.846 | 0.844 | 9000 |
| Weighted Avg | 0.864 | 0.847 | 0.845 | 9000 |

**Model Paths**
| Model | Path |
|-------|------|
| Model 1 | ./checkpoints/best_model_20241029_234652.pt |
| Model 2 | ./checkpoints/best_model_20241029_224507.pt |


## Visual Analysis of Results

### Confusion Matrix Insights
![Confusion Matrix](https://github.com/Ei3-kw/PatternAnalysis-2024/blob/topic-recognition/recognition/46822394_ViT_ADNC/img/confusion_matrix.png)

The confusion matrix reveals several key patterns:

1. **CN Classification (Top Row)**
   - Strong true negative rate: 4,344 correct CN identifications
   - Relatively low false positives: 196 MCI cases wrongly classified as CN
   - Shows model's strength in identifying normal cases

2. **MCI Classification (Bottom Row)**
   - Notable true positive rate: 3,275 correct MCI identifications
   - Higher false negatives: 1,185 CN cases misclassified as MCI
   - Indicates more conservative MCI detection

### ROC Curve Analysis
![ROC Curve](https://github.com/Ei3-kw/PatternAnalysis-2024/blob/topic-recognition/recognition/46822394_ViT_ADNC/img/roc_curves.png)

The ROC curves provide compelling evidence of model performance:

1. **Overall Performance**
   - Both classes achieve AUC = 0.96, significantly above random classification (dashed line)
   - Curves show similar performance for both CN and MCI classification
   - Sharp initial rise indicates excellent discrimination at high confidence thresholds

2. **Curve Characteristics**
   - Nearly identical curves for CN and MCI suggest balanced performance
   - Strong early climb (0.0-0.2 FPR range) indicates high confidence predictions are very reliable
   - Plateaus around 0.95 TPR, showing diminishing returns in sensitivity gains

## Quantitative Performance Breakdown

### Class-Specific Metrics
| Class | Key Strengths | Areas for Improvement |
|-------|---------------|----------------------|
| CN | 95.7% Recall | 78.6% Precision |
| MCI | 94.4% Precision | 73.4% Recall |

### Overall Performance Metrics
- **Accuracy**: 84.7%
- **Macro Average F1**: 0.844
- **Weighted Average F1**: 0.845

## Clinical Implications

1. **Screening Utility**
   - High CN recall (95.7%) makes it reliable for ruling out cognitive impairment
   - High MCI precision (94.4%) suggests confident positive predictions are trustworthy

2. **Risk Assessment**
   - False negative rate for MCI (26.6%) suggests need for additional verification of CN predictions
   - Could serve as an effective initial screening tool, with positive cases requiring clinical confirmation

## Recommendations

1. **Model Application**
   - Best suited for initial screening where high sensitivity to CN cases is desired
   - Consider threshold adjustments to balance precision/recall based on clinical priorities

2. **Future Improvements**
   - Focus on reducing CN false positives without sacrificing MCI detection if possible
   - Consider additional features or data augmentation to improve MCI recall
   - Investigate cases in the overlap region to identify potential distinguishing features

## License

Apache License - Version 2.0, January 2004 (http://www.apache.org/licenses/)

## Acknowledgments

- ADNI for providing the dataset https://adni.loni.usc.edu/
- Vision Transformer (ViT) original implementation
