import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone -> our featrue extractor

            Parameters:
                    backbone (str): Networks from https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a feature extractor network from the pretrained models
        self.backbone = models.__dict__[backbone](weights=ResNet18_Weights.DEFAULT, progress=True)

        # Get the number of features that are outputted by the last layer of feature extractor network
        out_features = list(self.backbone.modules())[-1].out_features

        # Our classification head with be an MLP with dense layers
        # The classification head classifies if the given combined feature
        # vector represents both malignant or both benign (same class of image) -> will return a value close to 1
        # Else if the images are of different classes, we want the head to return a value close to 0
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3*out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        '''
        Returns the similarity value between two images.

            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass the both images through the backbone network to get their separate feature vectors
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # Multiply (element-wise) the feature vectors of the two images together,
        # to generate a combined feature vector representing the similarity between the two.
        combined_features = feat1 * feat2

        # Combine this similarity vector with the original 2 feature vectors to pass to the Siamese net
        # This gives the dense Siamese layers the most opportunity to learn all patterns within the feature vectors
        final_vector = torch.cat((feat1, feat2, combined_features), dim=1)

        # Flatten the 3xN tensor to 1x(3*N) -> the MLP requires a flat input vector
        #flattened_combined = final_vector.view(-1)
        flattened_combined = final_vector.view(final_vector.size(0), -1)

        # Pass the final feature vector through classification head to get similarity value in the range of 0 to 1.
        output = self.cls_head(flattened_combined)
        return output
