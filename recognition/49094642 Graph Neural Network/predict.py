import torch
from modules import GCN
from dataset import DataLoader

def predict():
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"

    data_loader = DataLoader(edge_path, features_path, target_path)
    data = data_loader.create_data()

    model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=len(torch.unique(data.y)))
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        print(f"Predictions: {pred[data.test_mask]}")
        print(f"Ground Truth: {data.y[data.test_mask]}")

if __name__ == "__main__":
    predict()

