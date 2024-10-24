import torch
import matplotlib.pyplot as plt
from modules import GCN
from dataset import DataLoader

def train(model, data, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        return acc

def plot_metrics(losses, accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Over Epochs")
    plt.legend()

    plt.show()

def main():
    # Configuration parameters
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"

    data_loader = DataLoader(edge_path, features_path, target_path)
    data = data_loader.create_data()

    model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=len(torch.unique(data.y)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    losses = []
    accuracies = []

    for epoch in range(200):
        loss = train(model, data, optimizer, scheduler)
        acc = test(model, data)
        losses.append(loss)
        accuracies.append(acc)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
            
    plot_metrics(losses, accuracies)

    torch.save(model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    main()
