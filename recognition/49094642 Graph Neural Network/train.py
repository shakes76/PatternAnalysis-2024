import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

# Training function
def train(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item(), out

# Accuracy function
def accuracy(model, data, mask):
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = pred[mask].eq(data.y[mask]).sum().item()
        return correct / int(mask.sum())

# Train and evaluate the model
def train_and_evaluate(model, data, optimizer, class_weights, num_epochs=400):
    train_loss_values = []
    test_loss_values = []
    train_acc_values = []
    test_acc_values = []
    best_acc = 0
    best_train_acc = 0

    for epoch in range(num_epochs):
        train_loss, out = train(model, data, optimizer, class_weights)
        train_loss_values.append(train_loss)

        # Calculate training accuracy
        train_acc = accuracy(model, data, data.train_mask)
        train_acc_values.append(train_acc)

        # Calculate test loss and accuracy
        model.eval()
        test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask], weight=class_weights).item()
        test_loss_values.append(test_loss)

        test_acc = accuracy(model, data, data.test_mask)
        test_acc_values.append(test_acc)

        # Keep track of best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_train_acc = train_acc

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch:3d}  Train Accuracy: {train_acc:.4f}  Train Loss: {train_loss:.4f}  Test Accuracy: {test_acc:.4f}  Test Loss: {test_loss:.4f}')

    return train_loss_values, test_loss_values, train_acc_values, test_acc_values, best_train_acc, best_acc
