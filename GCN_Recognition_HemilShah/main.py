from dataset import load_dataset
from preprocess import preprocess
from modules import GCN
from train import train
from predict import predict
from visualise import visualize
import torch

# Load and preprocess dataset
graph_data = preprocess(load_dataset())

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph_data = graph_data.to(device)

# Initialize and train the model
model = GCN(128, 64, len(graph_data.y.unique())).to(device)
train_mask = (torch.rand(len(graph_data.y)) < 0.8).to(device)
train(model, graph_data, train_mask)

# Predict and evaluate
predict(model, graph_data)

# Visualize embeddings
with torch.no_grad():
    _, embeddings = model(graph_data)
visualize(embeddings.cpu().numpy(), graph_data)
