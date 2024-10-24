import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from modules import GCN
from dataset import DataLoader

class Trainer:
    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
      
class Tester:
    def test(model,data):
        model.eval()
        with torch.no_grad():  
            pred = model(data).argmax(dim=1) 
            correct = (pred == data.y).sum().item()
            acc = correct / data.num_nodes 
            return acc
