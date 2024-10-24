class Trainer:
    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
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
            pred = model(data).argmax(dim=1) 
            correct = (pred == data.y).sum().item()
            acc = correct / data.num_nodes 
            return acc
