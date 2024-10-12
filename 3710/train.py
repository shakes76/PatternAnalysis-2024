import torch
from modules import GNN
from dataset import load_data

def train(data, model, optimizer, loss_fn, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == "__main__":
    # Load data
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Initialize model, optimizer, and loss function
    model = GNN(in_channels=data.num_features, out_channels=6) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train model
    train(data, model, optimizer, loss_fn)
    torch.save(model.state_dict(), "model.pth")