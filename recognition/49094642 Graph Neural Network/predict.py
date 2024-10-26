import torch
from dataset import load_data
from module import GCN


# Function to load and evaluate the trained model
def predict(model, data):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        pred = model(data).argmax(dim=1)  # Predict class labels
        return pred


if __name__ == "__main__":
    # Load data
    edges_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    labels_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"
    data, num_classes = load_data(edges_path, labels_path, features_path)

    # Load trained model
    model = GCN(num_features=128, hidden_dim=256, num_classes=num_classes)
    model.load_state_dict(torch.load("best_model.pth"))  # Load the saved model weights
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Predict and display the results
    pred = predict(model, data.to('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Predicted classes: ", pred)

