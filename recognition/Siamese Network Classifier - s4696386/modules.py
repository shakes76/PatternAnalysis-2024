import torch, torchvision

"""
AI and MT tools used to
 - Wrangle pytorch package (Gain an understanding of what tools are available for Siamese Networks)
 - Understand and manipulate torch.nn.Sequential
 - Understand and manipulate torchvision models (such as the resnet34 eventually used for feature extraction)
 - Explaining and interpreting errors (and suggesting possible fixes)
"""

# With support from:
# https://github.com/pytorch/examples/blob/main/siamese_network
class SiameseNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SiameseNetwork, self).__init__(*args, **kwargs)

        # allowed under https://edstem.org/au/courses/18266/discussion/2269791
        self.resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self._dropout = 0.3
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_in_features * 2, 1042),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self._dropout),
            torch.nn.Linear(1042, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )

        # Define activation and loss criterion
        self.activation = torch.nn.Sigmoid()
        self.loss_criterion = torch.nn.BCELoss()

        # initialize weights
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size(0), -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.activation(output)

        return output

# recommended under https://edstem.org/au/courses/18266/discussion/2303273
class Classifier(torch.nn.Module):
    def __init__(self, pretrained_model: SiameseNetwork, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)

        # Load the feature extractor from the Siamese network
        self.feature_extractor = pretrained_model.resnet
        self.fc_in_features = pretrained_model.fc_in_features

        # Freeze feature extractor to prevent screwing it up
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Identical architecture as the Siamese network 
        self._dropout = 0.3
        self.fc = pretrained_model.fc

        # Recreate first fc layer to accept fc_in_features features (no longer fc_in_features * 2)
        self.fc[0] = torch.nn.Linear(self.fc_in_features, 1042)
        # Initialise the weights on the new Linear layer
        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        self.fc[0].bias.data.fill_(0.01)

        # Define activation and loss criterion
        self.activation = torch.nn.Sigmoid()
        self.loss_criterion = torch.nn.BCELoss()

    def forward(self, x):
        # Extract features from the image
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Classify the extracted features (as benign or malignant, no similar or dissimilar)
        output = self.fc(features)

        # Apply the activation function
        return self.activation(output)