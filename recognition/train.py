from torch.utils.data import *
from dataset import *
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from utiles import *
from modules import *
from sklearn.manifold import TSNE

# Hyper-parameters
num_epochs = 5
hidden_layer = 64
classes = ["Politicians", "Governmental Organisations", "Television Shows", "Companies"]
learning_rate = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():

    train_loss = []
    train_accuray = []

    validation_loss = []
    validation_accuracy = []

    epoch_loss = 0
    epoch_accuracy = 0

    print("Training loop:")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[train_mask].to(device), label_train.to(device))
        accuracy = ((output.argmax(dim=1)[train_mask] == label_train).float()).mean()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_accuray.append(accuracy)
        epoch_loss = loss.item()
        epoch_accuracy = accuracy.item()
        # Validation
        model.eval()

        output = model(data.x, data.edge_index)
        accuracy = ((output.argmax(dim=1)[validation_mask] == label_validation).float()).mean()

        validation_loss.append(loss.item())
        validation_accuracy.append(accuracy)
            

        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}")
    print("Training Over")
    torch.save(model, "GCN.pth")
            
def test_model():
    # ----- Testing -----
    print("--- Testing ---")
    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index)
        accuracy = ((output.argmax(dim=1)[test_mask] == label_test).float()).mean()
    print(f"Test Accuracy: {100 * accuracy:.2f}%")
    plot_TSNE(output.cpu().numpy(), data.y)


def plot_TSNE(output, y_true):
    # Plotting t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    reduced_embeddings = tsne.fit_transform(output)

    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):  
        idx = y_true == i 
        plt.scatter(reduced_embeddings[idx, 0], 
                    reduced_embeddings[idx, 1], y_true=classes[i], alpha = 0.7)  
    plt.legend()
    plt.title("TSNE Plot")
    plt.savefig("TSNE_plot.png")


if __name__ == "__main__":
    # Loading up the dataset and applying custom augmentations
    data, masks = load_data()
    train_mask = masks[0]
    test_mask = masks[1]
    validation_mask = masks[2]

    number_of_features = data.x.shape[1]

    label_validation = data.y[validation_mask]
    label_test = data.y[test_mask]
    label_train = data.y[train_mask]

    # Creating an instance of the UNet to be trained
    model = GCN(in_channels = number_of_features, hidden_channels= hidden_layer, out_channels =  len(classes))
    model = model.to(device)

    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    train_model()
    test_model()
    






 