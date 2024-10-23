class Trainer:
    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss()
        loss.backward()
        optimizer.step()

    def train(self, epochs=50):
        for epoch in range(epochs):
        loss = self.train_epoch()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
      
class Tester:
    def __init__(self, model, data, device='cuda'):
        model.test()
        self.model = model
        self.data = data.to(device)
        self.device = device
        self.model.to(device)
    
    def test(model,data):
        model.eval()
        with torch.no_grad():  
            out = self.model(self.data)  
            pred = model(data).argmax(dim=1) 
            correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum() 
            acc = int(correct) / int(self.data.test_mask.sum())  
            return acc
