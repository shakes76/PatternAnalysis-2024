import os
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import get_data_loaders
from modules import get_model, get_loss
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime

# Make sure GPU is available, added print statement to check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
csv_file = 'ISIC_2020_Training_GroundTruth_v2.csv'
img_dir = 'data/ISIC_2020_Training_JPEG/train/'

# Premilimary hyperparameters
batch_size = 32
embedding_dim = 64
learning_rate = 1e-3
num_epochs = 50

# Data loading
train_loader, test_loader = get_data_loaders(csv_file=csv_file, img_dir=img_dir, batch_size=batch_size)

# Siamese Initialization
model = get_model(embedding_dim=embedding_dim).to(device)
triplet_loss = get_loss(margin=1.0).to(device)
classifier_loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Storing losses and accuracies
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    runningLoss = 0.0

    for i, (anchor, positive, negative, labels) in enumerate(train_loader):
        print("Entered Loop batch number: ", i)
        if i > 10:
            break
        anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass stage
        anchor_out, positive_out, negative_out = model(anchor, positive, negative)
        tripletLoss = triplet_loss(anchor_out, positive_out, negative_out)
        classifierLoss = classifier_loss(model.classify(anchor), labels)
        loss = tripletLoss + classifierLoss
        # Backward pass stage
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

    # Calculate training loss and accuracy
    epoch_loss = runningLoss / len(train_loader)
    train_losses.append(epoch_loss)
    

    # Validation
    model.eval()
    validationRunningLoss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_anchor, val_positive, val_negative, val_labels in test_loader:
            val_anchor, val_positive, val_negative, val_labels = val_anchor.to(device), val_positive.to(device), val_negative.to(device), val_labels.to(device)
            val_anchor_out, val_positive_out, val_negative_out = model(val_anchor, val_positive, val_negative)
            val_triplet_loss = triplet_loss(val_anchor_out, val_positive_out, val_negative_out)
            val_classifier_loss = classifier_loss(model.classify(val_anchor), val_labels)
            val_loss = val_triplet_loss + val_classifier_loss
            validationRunningLoss += val_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(model.classify(val_anchor).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
    
    epoch_val_loss = validationRunningLoss / len(test_loader)
    test_losses.append(epoch_val_loss)
    epoch_val_accuracy = 100 * (correct / total)
    test_accuracies.append(epoch_val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_val_loss:.4f}, Test Accuracy: {epoch_val_accuracy:.2f}%")

    # Learning rate scheduler
    scheduler.step(epoch_val_loss)



current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_directory = f"runs/{current_time}"
os.makedirs(run_directory, exist_ok=True)

def save_plot(plt, file):
    plt.savefig(os.path.join(run_directory, file))
    plt.close()


# Save the model
modelPath = os.path.join(run_directory, 'siamese.pth')
torch.save(model.state_dict(), modelPath)
print(f"Model saved at {modelPath}")

# Plotting the losses
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
save_plot(plt, 'losses.png')

# Plotting the accuracies
plt.figure(figsize=(10,5))
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()
save_plot(plt, 'accuracy.png')

# Plotting the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(classes)) + 0.5, classes)
    plt.yticks(np.arange(len(classes)) + 0.5, classes)
    save_plot(plt, "confusion_matrix.png")

# Visualizing embeddings using t-SNE

def visualize_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for anchor, _, _, label in data_loader:
            anchor = anchor.to(device)
            embedding = model.get_embedding(anchor).cpu().numpy()
            embeddings.append(embedding)
            labels.extend(label.numpy())
    
    embeddings = np.concatenate(embeddings)
    tsne = TSNE(n_components=2, random_state=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plotting the embeddings
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='jet')
    plt.colorbar(scatter)
    plt.title('2D t-SNE Visualization of Embeddings')
    save_plot(plt, "embeddings.png")

visualize_embeddings(model, test_loader)

# Getting validation data for confusion matrix
model.eval()
predictions = []
labels = []

with torch.no_grad():
    for anchor, _, _, label in test_loader:
        anchor = anchor.to(device)
        output = model.classify(anchor)
        _, predicted = torch.max(output.data, 1)
        predictions.extend(predicted.cpu().numpy())
        labels.extend(label.cpu().numpy())

classes = ['benign', 'malignant']
plot_confusion_matrix(labels, predictions, classes)

