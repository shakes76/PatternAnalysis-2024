class CustomDataset(Dataset):
    def __init__(self, edge_path, data2_path, data3_pate, transform=None):
        self.edge_path = edge_path
        self.data2_path = data2_path
        self.data3_path = data3_path
      
    def load_edge(self):
        edge_def = pd.read_csv(self.edge_path)
        return torch.tensor(edge_df.values.T, dtype=torch.long)

    def data_data2(self):
        data2_def = pd.read_csv(self.edge_path)
        return torch.tensor(data2_df.values.T, dtype=torch.long)

    def data_data3(self):
        data3_def = pd.read_csv(self.edge_path)
        return torch.tensor(data3_df.values.T, dtype=torch.long)

    def create_data_loader(self, transform=None):
        edge = self.load_edges()
        data2 = self.load_edges()
        data3 = self.load_edges()
        dataset = Data(data_path, transform=transform)
        return data_loader
