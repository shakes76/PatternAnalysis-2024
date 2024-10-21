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
