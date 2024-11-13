# GCN Model for Facebook Dataset
**Author:** Hemil Nikesh Shah, 47851672.

## Project Overview

In this project, we develop a GCN model to classify nodes within a social network graph. The dataset contains features for each user (node), edges representing relationships, and target labels indicating the category of each node. The goal is to train the GCN to correctly predict these categories, utilizing graph structure and node-level features

## Algorithm Workflow

## 1. Dataset Loading:

Load the graph data, including node features, edges, and target labels.

## 2. Pre-processing:

Ensure that edges are properly formatted, and self-loops are added to stabilize training.

## 3. GCN Model Forward Pass:

Perform graph convolutions over node features using three GCN layers.

Use ReLU activations after the first two layers for non-linearity.

Output layer generates logits for multi-class classification.

## 4. Training:

Use Cross-Entropy Loss for multi-class classification.

Adam Optimizer with a learning rate of 0.01 and weight decay of 5e-4.

Train for 160 epochs and monitor the loss every 10 epochs.

## 5. Prediction and Evaluation:

Evaluate the model using accuracy score and classification report from sklearn.

Generate predictions by selecting the class with the highest logit value.

## 6. Visualization:

Use UMAP to visualize the learned embeddings and analyze cluster separation.

## 7. Model Architecture

The GCN model consists of 3 layers:

GCNConv 1: Input layer with size matching the number of input features (128)

GCNConv 2: Hidden layer with 64 hidden units

GCNConv 3: Output layer matching the number of unique classes

Activation Function: ReLU is applied after the first two layers to introduce non-linearity.

Optimizer: Adam with a learning rate of 0.01 and weight decay of 5e-4.

Loss Function: CrossEntropyLoss for multi-class classification.

## 8. Results and Performance

The model shows a steady decrease in loss, indicating successful learning. By the end of 160 epochs, the loss converges to 0.1093.

![Epoch Iterations with Decreasing Loss](Epoch_Iterations.png)


## 9. Evaluation Results

The model achieves high accuracy on the node classification task. Below are the performance metrics:

Overall Accuracy: 96%

Macro Average: Precision = 0.96, Recall = 0.95, F1-Score = 0.95
Weighted Average: Precision = 0.96, Recall = 0.96, F1-Score = 0.96

![Evaluation Results](Evaluation_results.png)

The results indicate that the model performs well across all classes with minimal variance in precision, recall, and F1-score.

## 10. Embedding Visualization

Using UMAP, we project the high-dimensional embeddings into a 2D space for visualization.This visualization helps confirm that the model has learned meaningful representations for the node categories.

![Embedding Visualization](Umap_Visualisation.png)


## References:

Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in Neural Information Processing Systems, 30, 1024–1034.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980. https://doi.org/10.48550/arXiv.1412.6980

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907. https://doi.org/10.48550/arXiv.1609.02907

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101. https://doi.org/10.48550/arXiv.1711.05101

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426. https://doi.org/10.48550/arXiv.1802.03426

Newman, M. E. J. (2018). Networks. Oxford University Press.

Powers, D. M. W. (2011). Evaluation: From precision, recall and F-measure to ROC, informedness, markedness & correlation. Journal of Machine Learning Technologies, 2(1), 37–63.

Rossi, R. A., & Ahmed, N. K. (2015). The network data repository with interactive graph analytics and visualization. Proceedings of the AAAI Conference on Artificial Intelligence, 4292–4293. https://doi.org/10.1609/aaai.v29i1.9363

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Wasserman, S., & Faust, K. (1994). Social network analysis: Methods and applications. Cambridge University Press.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Philip, S. Y. (2020). A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems, 32(1), 4–24. https://doi.org/10.1109/TNNLS.2020.2978386