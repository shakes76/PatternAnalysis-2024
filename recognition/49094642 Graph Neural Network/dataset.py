class CustomDataset(Dataset):
  def __init__(self, data_path, transform=None):
    self.data_path = data_path

  def data_data1(self):
    data1_def = pd.read_csv(self.edge_path)
    return torch.tensor(data1_df.values.T, dtype=torch.long)

  def data_data2(self):
    data2_def = pd.read_csv(self.edge_path)
    return torch.tensor(data2_df.values.T, dtype=torch.long)

  def data_data3(self):
    data3_def = pd.read_csv(self.edge_path)
    return torch.tensor(data3_df.values.T, dtype=torch.long)

  def create_data_loader(self, transform=None):
    data1 = self.load_edges()
    data2 = self.load_edges()
    data3 = self.load_edges()
    dataset = Dataset(data_path, transform=transform)
    return data_loader
