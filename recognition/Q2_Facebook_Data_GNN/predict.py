import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from modules import GNNModel, AdvanceGNNModel, AdvanceGATModel
from dataset import load_facebook_data, split_data
from train import train_model, test_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def predict(data, model):
    """
    Predict the labels for the given(GNN) data and model.
   
    This function predicts the labels for the given data and model.
    It loads the model, performs the forward pass and returns the predictions.
   
    Parameters:
    -----------
    data : torch_geometric.data.Data
        The data object containing the node features and edge index
    model : torch.nn.Module
        The model to use for prediction
       
    Returns:
    --------
    predictions : torch.tensor
        The predicted labels for the data
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    return pred

def visualise_results(data, pred):
    """
    Visualise the results of the prediction.
   
    This function visualises the results of the prediction by plotting the
    confusion matrix and classification report.
   
    Parameters:
    -----------
    data : torch_geometric.data.Data
        The data object containing the node features and edge index
    pred : torch.tensor
        The predicted labels for the data

    Returns:
    --------
    None
    """
    # Get the true labels
    true = data.y[data.test_mask].cpu().numpy()
    predicted_labels = pred[data.test_mask].cpu().numpy()

    # give classifications to the labels
    # confusion matrix
    cm = confusion_matrix(true, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels =['tvshow', 'government', 'company', 'politician'], 
                yticklabels =['tvshow', 'government', 'company', 'politician'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix - test data")
    print("Confusion Matrix:")
    plt.show()

    # classification report
    print("Classification Report:")
    print(classification_report(true, predicted_labels, 
                                target_names=['tvshow', 'government', 'company', 'politician']))


if __name__ == '__main__':
    """
    This main function predicts the labels for the Facebook dataset.
    The data is loaded using the load_facebook_data function from the dataset module.
    The model is loaded using the GNNModel class from the modules module.
    The model is used to predict the labels for the data.
    The results are visualised using the visualise_results function.
    The model is tested using the test_model function.
    """

    # File paths
    path = "recognition/Q2_Facebook_Data/facebook_large"
    features_path = f"{path}/musae_facebook_features.json"
    edges_path = f"{path}/musae_facebook_edges.csv"
    target_path = f"{path}/musae_facebook_target.csv"

    # load the data
    data = load_facebook_data(features_path, edges_path, target_path)
    # split the data
    train_mask, val_mask, test_mask = split_data(data, train_size=0.8, val_size=0.1)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # initialize the model
    # each model is test to see which one gives the best results
    
    # GNN model - not in use - only learn from one predict class
    # model = GNNModel(input_dim=128, hidden_dim=64, output_dim=4, num_layers=3) 

    # Advance GNN model 
    model = AdvanceGNNModel(input_dim=128, hidden_dim=[512])

    # GAT model - not in use - not much difference in the results compared to the AdvanceGNNModel
    # model = AdvanceGATModel(input_dim=128, hidden_dim=[128,128])

    # path2 =  "recognition/Q2_Facebook_Data/modelAdvanced2.pth"
    path1 =  "recognition/Q2_Facebook_Data/modelAdvanced1.pth"
    # Load trained model weights
    model.load_state_dict(torch.load(path1))
    
    # predict the labels
    pred = predict(data, model)

    # visualise the results
    visualise_results(data, pred)

    # test the model
    test_model(data, model)
    pass
