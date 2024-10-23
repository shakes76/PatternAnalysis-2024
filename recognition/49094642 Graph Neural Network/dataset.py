class CustomDataset(Dataset):
  def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    feature = self.data.iloc[idx, :-1].values 
    label = self.data.iloc[idx, -1]  
    feature = torch.tensor(feature, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)
    
    if self.transform:
      feature = self.transform(feature)
      return feature, label

def create_data_loader(data_path, batch_size=32, shuffle=True, transform=None):
  dataset = CustomDataset(data_path, transform=transform)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return data_loader
