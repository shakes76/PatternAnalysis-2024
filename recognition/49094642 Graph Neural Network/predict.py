import torch
from modules import GCN
from dataset import prepare_data, create_data


# Function to load the model and make predictions
def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)
        return predictions


if __name__ == "__main__":
    # File paths for loading data
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"

    # Load and prepare data
    node_features, edge_index, y, labels = prepare_data(edge_path, target_path, features_path)
    data = create_data(node_features, edge_index, y)

    # Load the model and weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=len(labels['page_type'].unique())).to(device)
    model.load_state_dict(torch.load('gcn_model.pth'))
    model = model.to(device)
    data = data.to(device)

    # Make predictions
    predictions = predict(model, data)

    # Print predictions (or visualize results as needed)
    print("Predictions: ", predictions.cpu().numpy())
