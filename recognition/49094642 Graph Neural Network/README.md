Graph Neural Network for Classification on the Facebook Large Page-Page Network

Author
Zhe Wu
student ID:49094642

Project Overview
a description of the algorithm and the problem that it solves

Version Transformer

Environment Dependencies
The project requires the installation of the following software or packages:
·Python 3.12.4
·Pytorch 2.4.1
·Cuda 11.8 
·Numpy 1.26.4
·scikit-learn 1.5.1
·Pandas 2.2.2
·Torch Geometric 2.6.1
·UMAP 0.1.1
·Matplotlib 3.9.2

 Reproduciblility Of Results
Setting the random seed for PyTorch: torch.manual_seed(42)

Repository




Data Set
This project uses the Facebook Large Page-Page Network dataset provided by the course. The dataset initially did not have a clear division into training and test sets. The facebook_edges dataset contains the relationship or connection information between different nodes (representing Facebook pages), indicating the connections in the network. The facebook_target dataset contains the target label for each node, representing different types of Facebook pages. The facebook_features dataset contains the features of each node (Facebook page) in the network. Each node is represented by an ID, and its corresponding feature (usually a numeric array) may describe the characteristics or activities of the page. I generate a random number for each node, and nodes with random numbers less than 0.8 are marked as training sets, and nodes that are not selected as training sets are used as test sets. That is, it is divided into 80% training sets and 20% test sets. This is done to ensure the randomness of the division, which does not depend on the specific attributes or order of the nodes, and ensures that there will be no deviation due to the order of data when training the model. At the same time, through random division, the generalization ability of the model on unknown data can be tested. The data layout is shown in the figure below:

![图片1](https://github.com/user-attachments/assets/79ec6647-d5e5-4976-a11d-a7d369ac2f81)
![image](https://github.com/user-attachments/assets/b9e71cbd-d5df-4175-a493-619376d6b5e8)
![image](https://github.com/user-attachments/assets/5808e22c-26ee-4f23-809e-eda2c395a3c1)

Result
![image](https://github.com/user-attachments/assets/0d1c557c-12fb-4610-aa6c-b4250a7e981b)
![image](https://github.com/user-attachments/assets/bc1f734a-fe17-4f0f-8790-c71289144233)
![image](https://github.com/user-attachments/assets/aee19182-2bfd-42cd-ad1c-214ac22f03e0)




