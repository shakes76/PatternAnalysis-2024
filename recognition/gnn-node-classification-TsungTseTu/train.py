import torch
from modules import GCN  # Import your GNN model
from dataset import load_facebook_data  # Load the dataset

print("script start")

def train():
    print("start training section")
    # Load data
    print("Loading data...")
    data = load_facebook_data('/data/facebook.npz')

    # Check if data loaded correctly
    print(f"Data successfully loaded. Number of nodes: {data.num_nodes}, Number of classes: {data.num_classes}")

    # Initialize the model
    model = GCN(input_dim=128, hidden_dim=64, output_dim=data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("start training...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        #For debugging: Print output shape
        print(f"Epoch {Epoch+1}, Output shape: {out.shape}")

        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # print the loss for each epoch
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the trained model without training it again
    torch.save(model.state_dict(), 'gnn_model.pth')
    print("Model saved to gnn_model.pth")

if __name__ == '__main_':                                                                     
    train()

