import torch
from modules import GNN
from dataset import load_data
from sklearn.metrics import accuracy_score

def train(data, model, optimizer, loss_fn, epochs=100):
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation phase
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask])
                val_predictions = out[data.val_mask].argmax(dim=1)
                val_accuracy = accuracy_score(data.y[data.val_mask].cpu(), val_predictions.cpu())
                
            print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy * 100:.2f}%')
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "model.pth")
                
if __name__ == "__main__":
    # Load data
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Initialize model, optimizer, and loss function
    model = GNN(in_channels=data.num_features, out_channels=4) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train model
    train(data, model, optimizer, loss_fn)
    print(f"Number of training nodes: {data.train_mask.sum().item()}")

