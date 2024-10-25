import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    device = torch.device("mps")# if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Lists to store the loss values for each epoch
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train() 
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()  
            optimizer.step() 
            
            train_loss += loss.item() * x.size(0)  
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)  # store train loss
        
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

        # Validation 
        model.eval()  
        val_loss = 0.0
        with torch.no_grad():  
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0) 

        # calculate validation loss for the epoch
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)  # store loss
        
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    plot_loss_curve(train_losses, val_losses)
    torch.save(model.state_dict(), '/Users/gghollyd/comp3710/report/module_weights.pth')

#plot this in train.py
def plot_loss_curve(train_losses, val_losses):
    epochs = len(train_losses)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

