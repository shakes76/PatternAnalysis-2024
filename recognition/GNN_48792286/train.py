#import necessary packages
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import the GNN class from a module named 'modules'
from modules import GNN
from dataset import load_data

#train the model
def train(model, data, epochs=200):
    # Initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()  

    # Store loss and accuracy for the plotting
    losses = []
    accuracies = []

    # Loop through each epoch
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, pred = out[data.train_mask].max(dim=1)  
        correct = pred.eq(data.y[data.train_mask]).sum().item() 
        acc = correct / data.train_mask.sum().item()

        # Record loss and accuracy
        losses.append(loss.item())  
        accuracies.append(acc)  

        # Print epoch, loss, and accuracy
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')
    # Return the recorded losses and accuracies
    return losses, accuracies  
# Function to plot loss and accuracy
def plot_loss_accuracy(losses, accuracies, epochs):
    fig, ax1 = plt.subplots()  

    ax1.set_xlabel('Epochs')  
    ax1.set_ylabel('Loss', color='tab:blue') 
    ax1.plot(range(1, epochs + 1), losses, color='tab:blue', label='Loss')  
    ax1.tick_params(axis='y', labelcolor='tab:blue')  

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Accuracy', color='tab:red')  
    ax2.plot(range(1, epochs + 1), accuracies, color='tab:red', label='Accuracy') 
    ax2.tick_params(axis='y', labelcolor='tab:red')  

    plt.title('Loss and Accuracy over Epochs')  
    ax1.legend(loc='upper left')  
    ax2.legend(loc='upper right')  
    plt.show() 

# Main function to run training
def main():
    # Load the dataset
    data_path = r"D:\HuaweiMoveData\Users\HUAWEI\Desktop\facebook.npz"  
    graph_data = load_data(data_path)  

    # Initialize model
    input_dim = graph_data.x.size(1)  
    hidden_dim = 64  
    output_dim = len(torch.unique(graph_data.y))  
    model = GNN(input_dim, hidden_dim, output_dim)  

    # Train the model
    losses, accuracies = train(model, graph_data)


    plot_loss_accuracy(losses, accuracies, epochs=200)


    torch.save(model.state_dict(), "trained_gnn_model.pth")

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
