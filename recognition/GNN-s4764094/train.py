from dataset import upload_dataset

tensor_edges, tensor_targets, tensor_features = upload_dataset()

# Check the results of those tensors
print("Data of edges: \n", tensor_edges)
print("Data of targets: \n", tensor_targets)
print("Data of features: \n", tensor_features)