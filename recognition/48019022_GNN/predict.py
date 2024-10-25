import torch
from dataset import GNNDataLoader
from modules import *

# Load data
data, train_idx, valid_idx, test_idx  = GNNDataLoader('/Users/anthonyngo/Documents/UQ/24sem2/COMP3710/project/PatternAnalysis-2024/facebook.npz')

# Initialize model
# Initialize model
architecture = "GAT"

if architecture == "GCN":
    # Select GCN
    model = GCNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1) # +1 as labels start from 0
elif architecture == "GAT":
    # Select GAT
    model = GATModelBasic(input_dim=128, hidden_dim=64, output_dim=data.y.max().item()+1)
elif architecture == "SAGE":
    model = GraphSAGE(input_dim=128, hidden_dim=64, output_dim=data.y.max().item()+1)
elif architecture == "SGC":
    model = SGCModel(input_dim=128, output_dim=data.y.max().item()+1, k=2)

savedpath = "best_" + architecture + "_model.pth"

model.load_state_dict(torch.load(savedpath))

def predict():
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    return pred

pred = predict()
print("Predictions:", pred)