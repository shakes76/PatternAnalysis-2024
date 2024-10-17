import torch
import torch.nn as nn
from torchvision import models

class FeatureExtraction(nn.Module):
    #CHANGE PRETRAINED TO TRUE IF IT SUCKS
    def __init__(self, embedding_dim = 256):
        "Init feature extractor using Resnet18 for our siamese networks"

        super(FeatureExtraction, self).__init__()

        #Load resnet
        self.model = models.resnet18(pretrained=False)

        #remove fully connected
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.embedding = nn.Linear(512, embedding_dim)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):


        features = self.model(x) #[batch_Size, 512, 1 ,1]
        features = features.view(features.size(0), -1) #[batch_Size, 512]
        embeddings = self.embedding(features) #[batch_size, embedding_dim]
        

        return embeddings
    
    

    