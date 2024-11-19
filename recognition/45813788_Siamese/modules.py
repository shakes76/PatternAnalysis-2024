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
            nn.Dropout(p=0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            
        )
    
 
        #true to train all layers false if not
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):


        features = self.model(x) #[batch_Size, 2048, 1 ,1]
        embeddings = self.embedding(features) #[batch_size, embedding_dim]
        
        logits = self.classifier(embeddings)

        return embeddings, logits
    
class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()

        self.feature_extractor = FeatureExtraction()

    def forward(self, x):

        embeddings,logits = self.feature_extractor(x)

        return embeddings, logits
