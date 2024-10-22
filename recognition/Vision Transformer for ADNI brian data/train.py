import dataset
import modules
import predict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader


### Model hyperparams
EPOCHS = 100
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-6
LR_EPOCHS = 20
WEIGHT_DECAY = 5e-4
EARLY_STOP = 10
BATCH_SIZE = 32


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # load data
    train_images, num_traing_points, val_images = dataset.load_data() 
    test_images, num_test_points, _ = dataset.load_data(testing=True)
    train_loader = DataLoader(train_images, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_images, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_images, batch_size=BATCH_SIZE)
    
    # define and instansiate model
    model = modules.GFNet(embed_dim=384, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.adamw(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_MAX=LR_EPOCHS, eta_min=LEARNING_RATE_MIN)
    # define vars 
    # prepare and complete model training
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    best_val_loss = float('inf')
    bad_val_count = 0

    current_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        
        current_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = torch.Tensor(labels).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct/total
        avg_train_loss = current_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for images, lables in val_loader:
                val_image = images.to(device)
                val_lables = torch.Tensor(labels).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100*val_correct/val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        current_epoch += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_val_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            bad_val_count += 1
        if bad_val_count >= EARLY_STOP:
            print(f"Early stopping at {current_epoch}")
            print(f"Final Train Loss: {avg_train_loss:.4f} & Acc: {train_acc:.2f}%")
            print(f"Final Val Loss: {avg_val_loss:.4f} & Acc: {val_acc:.2f}%")
            break
        
        if current_epoch == 1 or current_epoch % 20 == 0:
            print(f'Epoch [{current_epoch}/{EPOCHS}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_acc:.2f}%')

        scheduler.step(val_acc)

    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = torch.Tensor(labels).to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100* correct/total
    print(f'Test Acc: {test_acc:.2f}%')
    
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, current_epoch + 1), train_losses, label='Training Loss', marker='*')
    plt.plot(range(1, current_epoch + 1), val_losses, label='Validation Loss', marker='*')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss.png")  

    # Plotting training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, current_epoch + 1), train_accs, label='Training Accuracy', marker='x')
    plt.plot(range(1, current_epoch + 1), val_accs, label='Validation Accuracy', marker='x')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acc.png") 