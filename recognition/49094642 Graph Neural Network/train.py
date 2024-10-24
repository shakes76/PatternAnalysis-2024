import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight
from modules import GCN
from dataset import FacebookDatasetLoader
import numpy as np

# Trainer Class
class Trainer:
    def __init__(self, model, data, class_weights, optimizer):
        self.model = model
        self.data = data
        self.class_weights = class_weights
        self.optimizer = optimizer
        self.train_losses = []
        self.test_accuracies = []

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask], weight=self.class_weights)
        loss.backward()
        self.optimizer.step()
        self.train_losses.append(loss.item())
        return loss.item()

    def test(self):
        self.model.eval()
        pred = self.model(self.data).argmax(dim=1)
        correct = pred[self.data.test_mask].eq(self.data.y[self.data.test_mask]).sum().item()
        acc = correct / int(self.data.test_mask.sum())
        self.test_accuracies.append(acc)
        return acc

# Function to plot losses and accuracies
def plot_metrics(train_losses, test_accuracies):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(test_accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # File paths
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"
    feature_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"

    # Load dataset
    loader = FacebookDatasetLoader(edge_path, target_path, feature_path)
    node_features, edge_index, y, labels = loader.load_data()

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels['page_type']), y=labels['page_type'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data object
    data = Data(x=node_features, edge_index=edge_index, y=y)
    data.train_mask = torch.rand(data.num_nodes) < 0.8
    data.test_mask = ~data.train_mask

    # Set device and create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=len(labels['page_type'].unique())).to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    # Training loop
    trainer = Trainer(model, data, class_weights, optimizer)
    best_acc = 0
    patience = 20
    counter = 0

    for epoch in range(300):
        loss = trainer.train()
        test_acc = trainer.test()

        if test_acc > best_acc:
            best_acc = test_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Test Accuracy: {test_acc:.2f}')

    # Plot training metrics
    plot_metrics(trainer.train_losses, trainer.test_accuracies)
    print(f"Best Test Accuracy: {best_acc:.2f}")

if __name__ == "__main__":
    main()
    main()
