"""
Contains the source code for the components of the Siamese Net and Classifier.

Each component is implementated as a class or a function.
"""

###############################################################################
### Imports
import torch
import torch.nn as nn
from torchvision.models import resnet50


###############################################################################
### Config Settings
CONFIG = {
    'data_subset': 8000,
    'metadata_path': '/kaggle/input/isic-2020-jpg-256x256-resized/train-metadata.csv',
    'image_dir': '/kaggle/input/isic-2020-jpg-256x256-resized/train-image/image/',
    'embedding_dims': 128,
    'learning_rate': 0.0001,
    'epochs': 20,
}

###############################################################################
### Classes
class SiameseNet(nn.Module):
    def __init__(self, emb_dim=128):
        """
        """
        super(SiameseNet, self).__init__()

        # Load ResNet50 model
        resnet = resnet50() 
        
        # Slice out last fully connected layer so we can replace with our custom layers
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
               
        # Embedding Head
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
        
    def forward(self, img):
        """
        """
        out = self.feature_extractor(img)
        return self.fc_layers(out.view(out.size(0), -1))

    def get_embedding(self, img):
        """
        """
        return self.forward(img)
    
    def classify(self, img):
        """
        """
        return self.classifier(self.get_embedding(img))

class TripletLoss(nn.Module):
    """
    """
    def __init__(self, margin: float=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def euclidean_dist(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        """
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        """
        distance_positive = self.euclidean_dist(anchor, positive)
        distance_negative = self.euclidean_dist(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def set_seed(seed: int=42):
    """
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
