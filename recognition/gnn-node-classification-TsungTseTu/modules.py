import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

print("GCN class loaded")

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4,heads=4,dropout=0.2):
        super(GAT, self).__init__()


        # Define layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads)) #first GAT
        

        # only add if total layer is bigger than 2                  
        if num_layers > 2 :
            for _ in range(num_layers -2): #middle layers
                self.convs.append(GATConv(hidden_dim*heads, hidden_dim,heads=heads))

        self.convs.append(GATConv(hidden_dim*heads, output_dim,heads=1)) #Output

        # Batch Normalization layers
        if num_layers > 1:
            self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim*heads) for _ in range(num_layers -1)])

        self.dropout = torch.nn.Dropout(dropout) # Add dropout with 

    def forward(self, x, edge_index):
        # Print input shapes for debugging
        print(f"Input x shape: {x.shape}, edge_index shape: {edge_index.shape}")

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            print(f"Layer {i} conv output shape: {x.shape}")
            if i < len(self.bns):  # Apply batch norm for hidden layers
                x = self.bns[i](x)
                print(f"Layer {i} batch norm output shape: {x.shape}")
            x = F.relu(x)
            x = self.dropout(x)

        # Final layer (no ReLU)
        x = self.convs[-1](x, edge_index)
        print(f"Final layer output shape: {x.shape}")
        return x

