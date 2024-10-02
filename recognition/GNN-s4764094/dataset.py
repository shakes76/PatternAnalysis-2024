import pandas as pd
import torch
import json

# Check if the CUDA && MPS for our laptop is available
device = torch.cpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda is available.")
else:
    print("CPU usage.")

# Upload our dataset and transfer to tensor format, return those parameters
def upload_dataset():
    # Upload edge, target and feature
    edge_df = pd.read_csv("/Users/chenyihu/Desktop/Pycharm_Code/"
                          "3710-PatternAnalysis-2024/facebook_large/musae_facebook_edges.csv")

    target_df = pd.read_csv("/Users/chenyihu/Desktop/Pycharm_Code/"
                            "3710-PatternAnalysis-2024/facebook_large/musae_facebook_target.csv")

    with open("/Users/chenyihu/Desktop/Pycharm_Code/3710-PatternAnalysis-2024/facebook_large/"
              "musae_facebook_features.json", 'r') as f:
        feature_df = json.load(f)

    # Transfer and check whether the format is correct
    expected_length = 31
    target_df['page_name'] = pd.Categorical(target_df['page_name']).codes
    target_df['page_type'] = pd.Categorical(target_df['page_type']).codes

    tensor_edges = torch.tensor([edge_df['id_1'].values, edge_df['id_2'].values], dtype=torch.long).to(device)
    tensor_targets = torch.tensor(
        [target_df['id'].values, target_df['facebook_id'].values, target_df['page_name'].values,
         target_df['page_type'].values], dtype=torch.long).to(device)
    tensor_features = torch.tensor(
        [feature_df.get(str(node_id), [0] * expected_length)
         if len(feature_df[str(node_id)]) == expected_length
         else feature_df[str(node_id)] + [0] * (expected_length - len(feature_df[str(node_id)]))
         for node_id in target_df['id'].values],
        dtype=torch.float
    ).to(device)

    # Check the results of those tensors
    print("Data of edges: \n", tensor_edges)
    print("Data of targets: \n", tensor_targets)
    print("Data of features: \n", tensor_features)
    return tensor_edges, tensor_targets, tensor_features
