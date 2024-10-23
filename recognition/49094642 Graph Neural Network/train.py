class GNNTrainer:
    def train(model, data, optimizer):
        model.train()
        model.optimizer()
        out = model(data)
        loss = F.nll_loss()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, epochs=50):
        for epoch in range(epochs):
        loss = self.train_epoch()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
      
class GNNTester:
    def __init__(self, model, data, device='cuda'):
        model.test()
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
