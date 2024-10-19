import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class FeatureExtraction(nn.Module):
    #CHANGE PRETRAINED TO TRUE IF IT SUCKS
    def __init__(self, embedding_size = 256):
        "Init feature extractor using Resnet18 for our siamese networks"

        super(FeatureExtraction, self).__init__()

        #Load resnet
        self.model = models.resnet18(weights=None)#ResNet18_Weights.IMAGENET1K_V1)

        #remove fully connected
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_size),
            nn.ReLU(),
        )
 
        #true to train all layers false if not
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):


        features = self.model(x) #[batch_Size, 512, 1 ,1]
        embeddings = self.embedding(features) #[batch_size, embedding_dim]
        

        return embeddings
    
class SiameseNN(nn.Module):
    def __init__(self, embedding_size=256):
        super(SiameseNN, self).__init__()

        self.feature_extractor = FeatureExtraction(embedding_size)

    def forward(self, x):

        y = self.feature_extractor(x)

        return y

#classification network
class Classifier(nn.Module):
    
    def __init__(self, input_size=256, hidden_size=128):
        
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            #add dropout if needed
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        
        y = self.classifier(x)

        return y
    