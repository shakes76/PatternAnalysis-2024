import torch
from modules import GAT 
from sklearn.model_selection import train_test_split
from dataset import load_facebook_data  
from torch.nn import functional as F
import numpy as np

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

        # Split the data into training (70%) and testing (30%)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        # Convert split data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        edges= torch.tensor(edges, dtype=torch.long)

        # Re-index the edges for training set
        mask = (edges[:, 0]< X_train.size(0)) & (edges[:, 1] < X_train.size(0))
        edge_reindex = edges[mask].t()

        
        # Initialize the model
        print("Model starting...")
        input_dim = X_train.shape[1]
        output_dim = len(torch.unique(y_train)) #get number of unique class
        model = GAT(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layers=4, heads=4,dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = torch.nn.CrossEntropyLoss()

        # learn rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.3,patience=10)

        # Early stop
        early_stop_patience = 20
        early_stop_counter = 0
        best_loss = float("inf")

        # Training loop
        print("start training...")
        model.train()
        for epoch in range(10000):
            optimizer.zero_grad()
            out = model(X_train.clone().detach(), edge_reindex.clone().detach())
            loss = loss_fn(out,y_train)
            loss.backward()
            optimizer.step()

            # Learn rate scheduling
            scheduler.step(loss)

            # Early stop check
            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # If early stop counter hit patience threshold, stop training
            if early_stop_counter >= early_stop_patience:
                print(f"early stop at epoch {epoch+1}")
                break
            
            #print current l-rate and loss for each epoch
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, Learn rate: {current_lr}')

        # Save the trained model without training it again
        torch.save(model.state_dict(), 'gnn_model.pth', _use_new_zipfile_serialization=True)
        print("Model saved to gnn_model.pth")

    except Exception as e:
        print(f"An error occured: {e}")

if __name__ == '__main__':                                                                     
    train()

