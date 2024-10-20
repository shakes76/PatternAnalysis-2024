import torch.nn as nn
from torchvision import models

class FeatureExtraction(nn.Module):
    #CHANGE PRETRAINED TO TRUE IF IT SUCKS
    def __init__(self):
        "Init feature extractor using Resnet18 for our siamese networks"

        super(FeatureExtraction, self).__init__()

        #Load resnet
        self.model = models.resnet50(weights=None)

        #remove fully connected
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.embedding = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(p=0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)

        )
 
        #true to train all layers false if not
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):


        features = self.model(x) #[batch_Size, 2048, 1 ,1]
        embeddings = self.embedding(features) #[batch_size, embedding_dim]
        

        return embeddings
    
class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()

        self.feature_extractor = FeatureExtraction()

    def forward(self, x):

        y = self.feature_extractor(x)

        return y

#classification network
class Classifier(nn.Module):
    
    def __init__(self):
        
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            #add dropout if needed
            nn.Linear(64, 1),
            #nn.Sigmoid()
        )
        

    def forward(self, x):
        
        y = self.classifier(x)

        return y
    