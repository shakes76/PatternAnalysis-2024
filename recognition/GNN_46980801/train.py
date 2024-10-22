from modules import GCNNet
import torch
from dataset import graph_data
import matplotlib.pyplot as plt

#Training Loop 
def train(model, data):
    #Adam chosen for loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    #Cross Entropy Chosen for criterion 
    criterion = torch.nn.CrossEntropyLoss()

    #set ot training mode 
    model.train()
    
    optimizer.zero_grad()

    out = model(data)
    #Compute loss
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    #Gradient step 
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        #Pass the node data through the trained model 
        logits = model(data)
        #get predicitons 
        pred = logits.argmax(dim=1)
        
        # Compare the training predictions to the training labels 
        correct_train = (pred[data.train_mask] == data.y[data.train_mask]).sum()

        # Compare the test predictions to the test labels
        correct_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        
        #Report the proportion of correct train predictions 
        acc_train = int(correct_train) / int(data.train_mask.sum())
        
        #Report the proportion of correct test predictions 
        acc_test = int(correct_test) / int(data.test_mask.sum())

    return acc_train, acc_test

#Model definition 
#input dim, hidden dim to be tuned according ot training
model = GCNNet(input_dim=128, hidden_dim=64, output_dim=10)

#Configure epochs
epochs = range(1000)

losses = []
train_accs = []
test_accs = []

#Every epoch record and display key metrics (loss, train acc, test acc)
for epoch in epochs:

    loss = train(model, graph_data)
    
    acc_train, acc_test = evaluate(model, graph_data)
    losses.append(loss)
    train_accs.append(acc_train)
    test_accs.append(acc_test)
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train Acc: {acc_train:.4f}, Test Acc: {acc_test:.4f}')

#Save model weights
torch.save(model.state_dict(), 'weights.pth')

#Loss Plot
plt.figure()
plt.plot(epochs, losses, label="Losses")
plt.xlabel("Epochs")
plt.title("Loss Vs Epoch")
plt.ylabel("Loss")
plt.savefig('graphs/loss.png')

#Train accuracy plot
plt.figure()
plt.plot(epochs, train_accs, label="Training Accuracy")
plt.xlabel("Epochs")
plt.title("Training Accuracy Vs Epoch")
plt.ylabel("Accuracy")
plt.savefig('graphs/train_acc.png')


#test accuracy plot
plt.figure()
plt.plot(epochs, test_accs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.title("Test Accuracy Vs Epoch")
plt.ylabel("Accuracy")
plt.savefig('graphs/test_acc.png')


