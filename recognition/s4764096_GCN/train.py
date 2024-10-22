import torch as th
import torch.nn.functional as F
from dataset import prepare_data
from modules import GCN


def train():

    graph, train_mask, test_mask, in_feats = prepare_data()

    model = GCN(in_feats, num_classes=4)
    optimizer = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        logits = model(graph, graph.ndata['features'])

        loss = F.cross_entropy(logits[train_mask], graph.ndata['labels'][train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with th.no_grad():
            test_logits = model(graph, graph.ndata['features'])
            test_preds = test_logits[test_mask].argmax(dim=1)
            test_labels = graph.ndata['labels'][test_mask]
            accuracy = (test_preds == test_labels).float().mean()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Test Accuracy: {accuracy.item()}')

    th.save(model.state_dict(), "../../../_pycache_/gcn_model.pth")


if __name__ == '__main__':
    train()
