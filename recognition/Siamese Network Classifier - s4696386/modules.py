import torch, torchvision

# With support from:
# https://github.com/pytorch/examples/blob/main/siamese_network

class SiameseNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SiameseNetwork, self).__init__(*args, **kwargs)
        
        self.loss_criterion = torch.nn.BCELoss()
        
        self.resnet = torchvision.models.resnet18(weights=None)

        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_in_features * 2, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )

        self.sigmoid = torch.nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
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
        output = self.sigmoid(output)
        
        return output