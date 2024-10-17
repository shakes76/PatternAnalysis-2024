import numpy as np
import torch
from torch_geometric.data import Data


#Load in the facebook large page network dataset
data = np.load('facebook.npz')

#Explore the file names
#print("Data Files")
#print(data.files)

#Determine allocate each part of the data and determine the size 
edges = data['edges']  
#print("edge size")
#print(edges.shape)
features = data['features'] 
#print("Feature Size") 
#print(features.shape)
target = data['target']  
#print("Target Size")
#print(target.shape)

# create a tensor of edges, features and targets
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
X = torch.tensor(features, dtype=torch.float)
y = torch.tensor(target, dtype=torch.long)

#create train, validation, and test masks
num_nodes = X.shape[0]
train = torch.zeros( X.shape[0], dtype=torch.bool)

#70% SPlit 
train[:int(0.7 *  X.shape[0])] = True

val = torch.zeros( X.shape[0], dtype=torch.bool)
#10% For Validation 
val[int(0.7 *  X.shape[0]):int(0.8 *  X.shape[0])] = True

test = torch.zeros( X.shape[0], dtype=torch.bool)
#Remaining 20% for testing 
test[int(0.8 *  X.shape[0]):] = True

#create a data object using the pytorch-geometric data package 
graph_data = Data(x=X, edge_index=edge_index, y=y, train_mask=train, val_mask=val, test_mask=test)
