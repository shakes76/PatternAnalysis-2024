import torch.nn as nn
import torch as th
import dgl.nn as dglnn
import torch.nn.init as init


class GCN(nn.Module):
    def __init__(self, input_feats, num_classes=4):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.5)
        self.layers.append(dglnn.GraphConv(input_feats, 128, activation=nn.ReLU(), allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(128, 128, activation=nn.ReLU(), allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(128, 256, activation=nn.ReLU(), allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(256, 256, activation=nn.ReLU(), allow_zero_in_degree=True))
        self.classifier = nn.Linear(256, num_classes)
        self._initialize_weights()

    def forward(self, graph, inputs):
        x = inputs
        for layer in self.layers:
            x = self.dropout(layer(graph, x))
        return self.classifier(x)

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, dglnn.GraphConv):
                init.xavier_uniform_(layer.weight, gain=init.calculate_gain('relu'))
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        init.xavier_uniform_(self.classifier.weight, gain=init.calculate_gain('relu'))
        init.zeros_(self.classifier.bias)


