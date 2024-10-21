"""
File that contains the models that can be used by the classifier.
The currently supported model is ResNet50.

Made by Joshua Deadman
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50

# Code has been reworked from the following source:
#   https://github.com/sohaib023/siamese-pytorch/blob/master/siamese/siamese_network.py
class SiameseNetwork(nn.Module):
    """ A formal SiameseNetwork using resnet50 to extract features. """
    def __init__(self) -> None:
        """ Initialises the model and the feature extractor. """
        super().__init__()

        self._feature_extractor = resnet50(weights=None, progress=False)

    def forward_once(self, image) -> torch.Tensor:
        """ Performs a forward processing of a singular image.

        Arguments:
            image (torch.Tensor): A Tensor representing an image.
        Returns:
            A Tensor of the the extracted features of the image.
        """
        return self._feature_extractor(image)

    def forward(self, image1, image2, image3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Performs a forward processing of the three images.

        Arguments:
            image1 (torch.Tensor): A Tensor representing an image.
            image2 (torch.Tensor): A Tensor representing an image.
            image3 (torch.Tensor): A Tensor representing an image.
        Returns:
            A tuple of three Tensors. Each being the respective features of the image. 
        """
        return self.forward_once(image1), self.forward_once(image2), self.forward_once(image3)

class BinaryClassifier(nn.Module):
    """ Binary classifier to take the SiameseNetwork's features and provide a classification. """
    def __init__(self):
        """ Initialises an instance of the classifier. """
        super().__init__()
        self._layer1 = nn.Linear(1000, 500)
        self._layer2 = nn.Linear(500, 100)
        self._layer3 = nn.Linear(100, 2)
        self._reLU = nn.ReLU()
        self._activation = nn.Sigmoid()

    def forward(self, features) -> torch.Tensor:
        """ Performs a forward processing of the features. 

        Arguments:
            features (torch.Tensor): The features learnt in the SiameseNetwork. 
        Returns:
            The classification of a feature set.
        """
        x = self._layer1(features)
        x = self._reLU(x)
        x = self._layer2(x)
        x = self._reLU(x)
        x = self._layer3(x)
        return self._activation(x)



