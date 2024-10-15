# Multi-Class Node Classification of Facebook Network Dataset Using GCN

## Dataset information
This project uses the Facebook Large Page Network, which is a graph of pages on verified Facebook websites. The nodes represent official Facebook pages and the edges between nodes represent mutual likes between websites. The dataset is divided into 4 different categories: politicians, government organizations, TV shows and companies. The dataset consists of 22,470 nodes and 171,002 edges, each node has 128 features extracted from the website description created by the page owner.



## Algorithm

This project uses Graph Convolutional Network (GCN) for node classification, which is an algorithm that captures the dependencies between nodes through their connectivity. The model performs semi-supervised multi-class node classification on the Facebook dataset with the aim of categorizing each node into one of the four previously mentioned classes. Finally TSNE is used to visualize the data from high dimensional to two dimensional.


## Problem That It Solves

The problem solved in this project is the multi-category node classification problem, where each Facebook page is categorized into one of four categories.GCN solves the semi-supervised node classification problem using a small fraction of labeled nodes. The model is useful in a wide range of application scenarios, including social network analysis, advertisement targeting, and other tasks that require classification in complex graph structures.