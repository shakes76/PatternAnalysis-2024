import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from modules import GNNModel, AdvanceGNNModel, AdvanceGATModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import load_facebook_data, split_data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix 
import seaborn as sns
from sklearn.manifold import TSNE
import umap


def train_model(data, model, epochs=400, learning_rate=0.0012, weight_decay=2e-4, patience=20):
    """
    This function trains the given model on the given data.
    The training loop is run for the specified number of epochs.
    The model is trained using the Adam optimiser with the given learning rate.
    Different loss functions were used to train the model.
    1. basic cross entropy loss - not in use
    2. focal loss - in use
    The focal loss function is used to train the model.
    The learning rate is reduced by a factor of 0.4 if the validation accuracy does not improve for 7 epochs.
    The training loss and validation accuracy are tracked for each epoch and plotted at the end.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        The input data for the model
    model : torch.nn.Module
        The model to train
    epochs : int, optional (default=400)
        The number of epochs to train the model for
    learning_rate : float, optional (default=0.0012)
        The learning rate for the Adam optimiser
    weight_decay : float, optional (default=2e-4)
        The weight decay for the Adam optimiser
    patience : int, optional (default=20)
        The number of epochs to wait for an improvement in 
        validation accuracy before stopping training
   
    Returns:
    --------
    model : torch.nn.Module
        The trained model
    """
    # Prepare masks
    train_mask, val_mask, test_mask = split_data(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

    # basic cross entropy loss - not in use
    # loss_fn = torch.nn.CrossEntropyLoss()     

    # class weights
    class_weights = torch.tensor([3.0, 1.0, 1.8, 1.8])
    # reweight the classes
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # loss function - focal loss - not in use
    # loss_fn = FocusLoss()
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.4, patience=7, verbose=True)

    # Track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    # early stop setup
    best_val_acc = 0
    patience_counter = 0
    # training loop
    for epoch in range(epochs):
        model.train()
        # forward pass
        optimiser.zero_grad()
        out = model(data.x, data.edge_index)

        # loss
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

        # backward pass
        loss.backward()
        # optimiser step
        optimiser.step()

        # track losses
        train_losses.append(loss.item())

        # validation
        model.eval()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(correct.sum()) / int(data.val_mask.sum())
            val_accuracies.append(val_acc)

            ## track validation loss as well
            val_out = model(data.x, data.edge_index)
            val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask])
            val_loss = val_loss.item()
            val_losses.append(val_loss)

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # save the model
            torch.save(model.state_dict(), "recognition/Q2_Facebook_Data/modelAdvanced1.pth")

        else:
            patience_counter += 1

        # stop training if no improement in validation accuracy
        if patience_counter >= patience and val_acc>=max(val_accuracies):
            print(f"Early stopping at epoch {epoch}")
            break

        # learning rate scheduler
        scheduler.step(val_acc)
        # print (each value upto 4 deicmal places)
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

    # show confusion matrix on console only for both train, valida data
    correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(correct.sum()) / int(data.train_mask.sum())
    print(f"Train Accuracy: {train_acc:.4f}")

    # print confusion matrix to console
    train_preds = out[data.train_mask].argmax(dim=1).cpu().numpy()
    train_labels = data.y[data.train_mask].cpu().numpy()

    train_confusion_matrix = confusion_matrix(train_labels, train_preds)
    print("Confusion Matrix for Train Data:")
    print(train_confusion_matrix)

    # validation data
    correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(correct.sum()) / int(data.val_mask.sum())
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # print confusion matrix to console
    val_preds = out[data.val_mask].argmax(dim=1).cpu().numpy()
    val_labels = data.y[data.val_mask].cpu().numpy()

    val_confusion_matrix = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix for Validation Data:")
    print(val_confusion_matrix)

    # PLotting validation acc and training loss on separate graphs
    plt.figure(figsize=(10, 8))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    
    # save the graph
    plt.savefig('recognition/Q2_Facebook_Data/val_accuracy.png')
    plt.show()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Losses')
    # save the graph 
    plt.savefig('recognition/Q2_Facebook_Data/losses.png')
    plt.show()

    return model


def test_model(data, model):
    """
    This function tests the GNN model on the give data path.
    The test accuracy is printed to the console.
    The confusion matrix is printed to the console.
    The classification report is printed to the console.
    The confusion matrix is plotted and saved to a file.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        The input data for the model
    model : torch.nn.Module
        The trained model - GNN Model
   
    Returns:
    --------
    test_acc : float
        The test accuracy of the model        

    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(correct.sum()) / int(data.test_mask.sum())
        print(f"Test Accuracy: {test_acc:.4f}")

    return test_acc


# tsne embedding for visualisation
def tsne_embedding(data, model):
    """
    This function visualises the t-SNE embedding of the given data.
    The t-SNE embedding is computed using the t-SNE algorithm.
    The t-SNE embedding is plotted and displayed.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        The input data for the model
    model : torch.nn.Module
        The trained model - GNN Model

    Returns:
    --------
    None

    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    print(embeddings.shape)
    # tsne embedding 
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings2D = tsne.fit_transform(embeddings.cpu())

    # plot the t-SNE embedding
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings2D[:, 0], embeddings2D[:, 1], c=data.y, cmap='tab10', alpha=0.5)
    plt.colorbar()
    plt.title("t-SNE embedding  data")
    plt.show()

# simalrly UMAP embedding can be done as well
def umap_embedding(data, model):
    """
    This function visualises the UMAP embedding of the given data.
    The UMAP embedding is computed using the UMAP algorithm.
    The UMAP embedding is plotted and displayed.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        The input data for the model
    model : torch.nn.Module
        The trained model - GNN Model

    Returns:
    --------
    None
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    umap_embeddings = umap.UMAP(
        n_neighbors=50,
          min_dist=0.1, n_components=2, random_state=42).fit_transform(embeddings.cpu())

    # plot the UMAP embedding
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=data.y, cmap='tab10', alpha=0.7)
    plt.colorbar()
    plt.title("UMAP embedding data")
    plt.show()


class FocusLoss(torch.nn.Module):

    """
    This class implements the focal loss function to train the GNN model.
    The focal loss function is defined as:
    loss = (1-pt)^gamma * CE_loss
    where pt = exp(-CE_loss)
    CE_loss is the cross entropy loss.
    gamma is a hyperparameter that controls the focus of the loss function.

    """

    def __init__(self, weight=None, gamma=2):
        """
        This function initialises the focal loss function.
        The weight parameter is used to reweight the classes.
        The gamma parameter is used to control the focus of the loss function.
        
        Parameters:
        -----------
        weight : torch.tensor, optional (default=None)
            The class weights for the loss function
        gamma : float, optional (default=2)
            The gamma parameter for the focal loss function

        Returns:
        --------
        None
        """
        super(FocusLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        This function computes the focal loss for the given input and target.

        Parameters:
        -----------
        input : torch.tensor
            The input tensor from the model
        target : torch.tensor
            The target tensor for the model

        Returns:
        --------
        loss : torch.tensor
            The focal loss for the given input and target
        """
        ce_loss =  F.cross_entropy(input, target, weight=self.weight) 
        pt = torch.exp(-ce_loss)
        loss = (1-pt)**self.gamma * ce_loss
        return loss.mean()

if __name__ == '__main__':
    """
    This main function trains a GNN model on the Facebook dataset.
    The data is loaded using the load_facebook_data function from the dataset module.
    The model is trained using the train_model function(GNN).

    Three different models are implemented in this block to test the best model for the given dataset.
    1. Basic GNN model - not in use
    2. Advanced GNN model - model in use for training
    3. Advanced GAT model - not in use
    The best model is selected based on the performance of the model on the given dataset.
    Other models are commented out in the code.

    The trained model is tested using the test_model function.
    The test accuracy, confusion matrix and classification report are printed to the console.
    The confusion matrix is plotted and saved to a file.
    """
    # File paths
    path = "recognition/Q2_Facebook_Data/facebook_large"
    features_path = f"{path}/musae_facebook_features.json"
    edges_path = f"{path}/musae_facebook_edges.csv"
    target_path = f"{path}/musae_facebook_target.csv"

    # load the data
    data = load_facebook_data(features_path, edges_path, target_path)  

    # initialise the model
    # Basic GNN - not used
    # model = GNNModel(input_dim=128, hidden_dim=64, output_dim=4, num_layers=3)

    # Advance GNN - not used
    model = AdvanceGNNModel(input_dim=128, hidden_dim=[512])

    # Advance GAT 
    # model = AdvanceGATModel(input_dim=128, hidden_dim=[128,128])

    # train the model
    model = train_model(data, model)

    # test the model
    test_model(data, model)

    # tsne embedding
    tsne_embedding(data, model)

    # umap embedding
    umap_embedding(data, model)

    # save the model - not in use as model is saved in train_model function
    # modelName = "recognition/Q2_Facebook_Data/modelEnhance1.pth"
    # modelName = "recognition/Q2_Facebook_Data/modelAdvance1.pth"
    # modelName = "recognition/Q2_Facebook_Data/modelAdvance2.pth"
    # torch.save(model.state_dict(), modelName)