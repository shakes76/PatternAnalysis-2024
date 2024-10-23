class CustomDataset(Dataset):
    def __init__(self, edge_path, data2_path, target_path, transform=None):
        self.edge_path = edge_path
        self.data2_path = data2_path
        self.target_path = target_path
      
    def load_edge(self):
        edge_def = pd.read_csv(self.edge_path)
        return torch.tensor(edge_df.values.T, dtype=torch.long)

    def data_data2(self):
        data2_def = pd.read_csv(self.data2_path)
        return torch.tensor(data2_df.values.T, dtype=torch.long)

    def load_target(self):
        target_def = pd.read_csv(self.target_path)
        return torch.tensor(target_df['target'].values, dtype=torch.long)

    def create_data_loader(self, transform=None):
        edge = self.load_edges()
        x = self.load_data2()
        y = self.load_target()
        dataset = Data(data_path, transform=transform)
        return data_loader
