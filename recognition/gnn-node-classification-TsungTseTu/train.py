import torch
from modules import GCN  # Import GCN model
from sklearn.model_selection import train_test_split
from dataset import load_facebook_data  # Load the dataset
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
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(features, target, range(features.shape[0]), test_size=0.3, random_state=42)

        #Remap node indices in edges to match the reduced set of nodes in the training
        train_node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(train_indices)}
        train_mask = np.isin(edges[:, 0], train_indices) & np.isin(edges[:, 1],train_indices)
       
        #Filter edged to only include edges where both nodes are in training set
        edges_train = edges[train_mask]
        
        #Convert features, edges, and target to Pytorch tensors
        X_train = torch.tensor(X_train,dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        

        # Remap the edge indices based on the training set
        edge_list= []
        for edge in edges_train:
            if edge[0] in train_node_map and edge[1] in train_node_map:
                remapped_edge= [train_node_map[edge[0]], train_node_map[edge[1]]]
                edge_list.append(remapped_edge)
        

        edges_train = np.array(edge_list).T
        edges_train = torch.tensor(edges_train, dtype=torch.long)
        
        # Initialize the model
        print("Model starting...")
        input_dim = X_train.shape[1]
        output_dim = torch.unique(y_train).size(0) #get number of unique class
        model = GCN(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # learn rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=10, verbose=True)

        # Early stop
        early_stop_patience = 20
        early_stop_counter = 0
        best_loss = float("inf")

        # Training loop
        print("start training...")
        model.train()
        for epoch in range(500):
            optimizer.zero_grad()
            out = model(X_train, edges_train)
            loss = torch.nn.functional.nll_loss(out,y_train)
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
            
            # print the loss for each epoch
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Save the trained model without training it again
        torch.save(model.state_dict(), 'gnn_model.pth', _use_new_zipfile_serialization=True)
        print("Model saved to gnn_model.pth")

    except Exception as e:
        print(f"An error occured: {e}")

if __name__ == '__main__':                                                                     
    train()

