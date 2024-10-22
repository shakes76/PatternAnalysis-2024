"""
Contains the source code for the components of the Siamese Net and Classifier.

Each component is implementated as a class or a function.
"""

###############################################################################
### Imports
import torch
import torch.nn as nn
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, roc_auc_score


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
        # Since we are not predicing 1000 classes here - we need our own custom Feature Extractor Head
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
        
    def forward(self, img):
        """
        """
        out = self.feature_extractor(img)
        return self.fc_layers(out.view(out.size(0), -1))
    
    def classify(self, img):
        """
        """
        return self.classifier(self(img))

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