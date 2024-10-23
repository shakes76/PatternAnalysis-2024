class CustomDataset(Dataset):
    def __init__(self, edge_path, features_path, target_path, transform=None):
        self.edge_path = edge_path
        self.data2_path = data2_path
        self.target_path = target_path
      
    def load_edge(self):
        edge_df = pd.read_csv(self.edge_path)
        return torch.tensor(edge_df.values.T, dtype=torch.long)

    def load_features(self):
        features_df = pd.DataFrame(features_list)
        return torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float)
   
    def load_target(self):
        target_df = pd.read_csv(self.target_path)
        return torch.tensor(target_df['target'].values, dtype=torch.long)

    def create_data_loader(self, transform=None):
        edge = self.load_edges()
        x = self.load_features()
        y = self.load_target()
        dataset = Data(data_path, transform=transform)
        return data_loader
