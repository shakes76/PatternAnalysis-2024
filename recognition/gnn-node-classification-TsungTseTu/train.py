import torch
from modules import GCN  # Import your GNN model
from dataset import load_facebook_data  # Load the dataset

print("script start")

def train():
    try:
        print("start training section")

        # Load data
        print("Loading data...")
        data = load_facebook_data()
        
        if data is None:
            print("Data loading error. Exiting...")
            return

        # print available arrays in the dataset if successfully loaded
        print(f"Data successfully loaded. Available arrays:{data.files}")
    
        edges = data['edges']
        features = data['features']
        target = data['target']

        
        print(f"Edge shape: {edges.shape}, features shape: {features.shape}, target shapes: {target.shape}")

        # Initialize the model
        print("Model starting...")
        input = features.shape[1]
        output = len(set(target))
        model = GCN(input_dim=input, hidden_dim=64, output_dim=output)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        print("start training...")
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.functional.nll_loss(out,target)
            loss.backward()
            optimizer.step()

            # print the loss for each epoch
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Save the trained model without training it again
        torch.save(model.state_dict(), 'gnn_model.pth')
        print("Model saved to gnn_model.pth")

    except Exception as e:
        print(f"An error occured: {e}")

if __name__ == '__main__':                                                                     
    train()

