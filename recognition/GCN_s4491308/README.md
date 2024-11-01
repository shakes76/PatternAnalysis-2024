# Semi supervised multi-class node classification of Facebook Large Page-Page Network dataset using GCN
## Dataset Overview 
The Facebook Large Page - Page Network Dataset was used in this project. The dataset is a page-page graph of verified facebook sites. Nodes extracted from the site descriptions that the page owners created to summarize the purpose of the site represent the official facebook pages whereas the edges represent the links between them. The sites belong to four categories namely Politicians , Governmental Organisations, Television Shows and Companies. The dataset has 22,470 nodes and 171,002 edges. A preprocessed version of the dataset facebook.npz was used for the multiclass classification that contained features in the form of 128 dimensional vectors. 

## Algorithm Used 
This project utilises Graph Convolutional Network (GCN) to perform the semi supervised multiclass classicification of the facebook sites into the above mentioned categories. A t-SNE plot is also plotted to visualise the node embeddings learned by the model in two dimensions. 

## The Problem that it solves 
Graphical data is very insighful becuase it has the ability to capture the inherent structural relationships in the data and in hence widely used in various real-world application domains such social media analysis, medical imaging of molecular structures and transportation system. Traditional CNN's fail to work on graphical data that is where GCN's come in. The identification of classes in the facebook data using GCN can provide great insights into consumer behaviour like help in providing targetted recommendations to customers based on usage history etc.

## How it works 
GCN works by learning fatures by aggregating information from the neighbouring nodes. Each layer in the GCN performs convolution and updates the representation of a node based on its features as well as the features of the neighbours it is connected to. This project utilised [PyTorch Geometric GCN model](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html)

## Description of Architecture 





## Data pre processing steps 



## Design choices 




## Results 




## Conclusion 

## References 
Articles read to gain understanding of GCN and for ReadMe description: 
[Graph Convolutional Networks (GCN) Explained At High Level](https://towardsai.net/p/l/graph-convolutional-networks-gcn-explained-at-high-level)
[A Comprehensive Introduction to Graph Neural Networks (GNNs)](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial)

The model architecture and tSNE visualisation is based on code from [A Comprehensive Introduction to Graph Neural Networks (GNNs)](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial)

and uses the prebuilt [Pytorch Geometric GCN model for the layers](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html)

The data preprocessing code is inspired from [pytorch-geometric example](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/mask.py#L41)

The training and testing loops are inspired and adapted from code at [pytorch-gcn.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py#L81)
