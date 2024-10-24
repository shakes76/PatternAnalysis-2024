"""
Contains the source code for the components of the Siamese Net (Feature Extractor and Classifier).
Additionally contains the TripletLoss function that should be used to train the Siamese Net.
"""


###############################################################################
### Imports
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np


###############################################################################
### Classes
class SiameseNet(nn.Module):
    """
    Siamese Net model designed for classification of the ISIC 2020 data set.
    Split up into two components the Feature Extractor to creating embeddings of the data
    and Classifier to convert the embeddings into a class prediction.
    
    The embedding space will be of dimension 'emb_dim'.
    The Feature Extractor section of the model should ideally be trained with TripletLoss.
    """
    def __init__(self, emb_dim=128):
        super(SiameseNet, self).__init__()

        # Load ResNet50 model
        resnet = resnet50()
        
        # Slice out last fully connected layer so we can replace with our custom layers
        # Since we are not predicting 1000 classes here - we need our own custom Feature Extractor Head
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
               
        # Feature Extractor Head
        # Used to produce embeddings
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, emb_dim)
        )

        # Classifier Head
        # Used to convert embeddings into classification
        self.classifier = nn.Linear(emb_dim, 2)
        
    def forward(self, img: torch.tensor) -> torch.tensor:
        """
        Pass image through the Feature Extractor and return
        the embedding of the image.

        Returns: embedding of image
        """
        out = self.feature_extractor(img)
        return self.fc_layers(out.view(out.size(0), -1))
    
    def classify(self, img: torch.tensor) -> torch.tensor:
        """
        Pass the image through the Feature Extractor and then the
        Classifier to produce a classification of the image (class prediction)

        Returns: classification prediction of image
        """
        return self.classifier(self(img))

class TripletLoss(nn.Module):
    """
    Will take the triplets (in embedded form) and will calculate the loss via the following function.

    loss=max(0,D(A,P)-D(A,N)+margin)

    Where D represents Euclidean distance, A, P and N represent the output embeddings of the Anchor,
    Positive and Negative images from the triplet respectively, and margin is a hyper parameter to
    enforce a minimum separation between classes.

    i.e. this loss function will penalise data points of different classes being close
    to each other in the embedded space.
    """
    def __init__(self, margin: float=1.0) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin

    def euclidean_dist(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Returns: euclidean distance between x1 and x2
        """
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Calculates triplet loss via the function mentioned in the TripletLoss class
        doc string.

        Returns: triplet loss
        """
        distance_positive = self.euclidean_dist(anchor, positive)
        distance_negative = self.euclidean_dist(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


###############################################################################
### Config Settings
def get_config() -> dict:
    """
    Get the config used to train and evaluate SiameseNet on the ISIC 2020 data.

    Returns: the current config settings
    """
    config = {
        # If we wish to only use a subset of the data to training
        # Set to: None to use the full dataset.
        # Smaller numbers will speed up training but may reduce final model performance.
        'data_subset': 16000,
        'metadata_path': './data/train-metadata.csv',
        'image_dir': './data/train-image/image/',
        'batch_size': 32,
        'embedding_dims': 128,
        'learning_rate': 0.0001,
        'epochs': 20,
    }
    return config


###############################################################################
### Set Seed Helper
def set_seed(seed: int=42) -> None:
    """
    Sets the seed for a number of random number generators that will be used
    This helps to ensure reproducibility between runs
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
