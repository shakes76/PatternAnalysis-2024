import torch
from modules import GAT
from dataset import load_facebook_data  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

def visualize(embeddings, labels):
    tsne = TSNE(n_components=2,perplexity=30,learning_rate=200)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Spectral')
    plt.colorbar()
    plt.title("t-SNE visualization of GAT embedding")
    plt.show()

def predict():
    # Load data
    data = load_facebook_data()

    # Extract features, edges, and target
    features = torch.tensor(data['features'], dtype=torch.float32)
    edges = torch.tensor(data['edges'].T, dtype=torch.long)
    target = torch.tensor(data['target'], dtype=torch.long)
   
    # Load the split test data
    _, X_test, _, y_test = train_test_split(features,target,test_size=0.3,random_state=42)

    #re-index the dges for test set
    edge_reindex = edges[:, torch.all(edges<X_test.size(0), dim=0)]

    # Set input nad output based on feature and target
    input_dim = X_test.shape[1] #Number of features per node
    output_dim = len(torch.unique(y_test)) # number of unique classes

    # Initialize the model (same as in training)
    model = GAT(input_dim=input_dim, hidden_dim=512, output_dim=output_dim, num_layers=4, heads=4,dropout=0.3)

    # Load pre-trained model
    model.load_state_dict(torch.load("gnn_model.pth", weights_only=True))
                        
    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get predictions
    with torch.no_grad():
        out = model(X_test, edge_reindex) #raw output
        prediction = out.argmax(dim=1) #Get predicted class for each node
        # Check Accuracy
        correct = (prediction==y_test).sum().item() #Collect correctly predicted
        accuracy = correct/y_test.size(0)
        print(f"Accuracy:{accuracy*100:.2f}%")

    # Visualize embeddings using TSNE
    embeddings = out.numpy()
    labels = y_test.numpy()
    visualize(embeddings, labels)

if __name__ == '__main__':
    predict()
