class GNNTrainer:
  def __init__(self, model, optimizer, data, loss_fn=torch.nn.functional.nll_loss, device='cuda'):
    self.model = model
    self.optimizer = optimizer
    self.data = data.to(device)
    self.loss_fn = loss_fn
    self.device = device
    self.model.to(device)

  def train_epoch(self):
    self.model.train()
    self.optimizer.zero_grad()
    out = self.model(self.data)  
    loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])  
    loss.backward()  
    self.optimizer.step() 
    return loss.item()

  def train(self, epochs=50):
    for epoch in range(epochs):
      loss = self.train_epoch()
      print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
      
class GNNTester:
  def __init__(self, model, data, device='cuda'):
     self.model = model
     self.data = data.to(device)
     self.device = device
     self.model.to(device)
  def test(self):
     self.model.eval()
     with torch.no_grad():  
     out = self.model(self.data)  
     pred = out.argmax(dim=1)  
     correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum() 
     acc = int(correct) / int(self.data.test_mask.sum())  
     return acc
