class GNN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(GNN, self).__init__()
    self.conv1 = GCNConv(input_dim, hidden_dim)
    self.conv2 = GCNConv(hidden_dim, output_dim)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index 
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    F = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)
    return F.log_softmax(x, dim=1)  
