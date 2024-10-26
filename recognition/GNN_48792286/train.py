# Model training function with early stopping
def train(model, data, epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_loss = float('inf')
    patience_counter = 0

    # For storing loss and accuracy
    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Check for early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Calculate accuracy on training set
        _, pred = out[data.train_mask].max(dim=1)
        acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

        losses.append(loss.item())
        accuracies.append(acc)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

    return losses, accuracies
