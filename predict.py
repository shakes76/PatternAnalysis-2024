import torch
from dataset import load_facebook_data
from modules import GNNModel

# Load data
data = load_facebook_data('facebook.npz')

# Initialize model
model = GNNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)
model.load_state_dict(torch.load('gnn_model.pth'))

def predict():
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    return pred

pred = predict()
print("Predictions:", pred)
