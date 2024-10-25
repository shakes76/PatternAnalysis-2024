import dataset
import modules
import predict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

### Model hyperparams
EPOCHS = 50
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-6
LR_EPOCHS = 25
WEIGHT_DECAY = 5e-4
EARLY_STOP = 15
BATCH_SIZE = 32

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def train(model, dl, crit, opt, sched, tqdm_enabled=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dl, disable=tqdm_enabled):
        inputs, labels = inputs.to(device), labels.float().to(device)
        opt.zero_grad()

        outputs = model(inputs)
        loss = crit(outputs, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item() 
        pred = (outputs >= 0).float()
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    avg_loss = total_loss / len(dl)
    acc = 100 * correct / total

    if sched is not None:
        sched.step()

    return avg_loss, acc

def test(model, dl, crit, tqdm_enabled=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dl, disable=tqdm_enabled):
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs)
            loss = crit(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs >= 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dl)
    acc = 100 * correct / total
    
    return avg_loss, acc

def plot(train_accs, test_acc):
    epochs = range(1, EPOCHS+1)
    plt.figure(figsize=(18, 8))

    plt.plot(epochs, train_accs, label="Average Training Accuracy per Epoch", colour="blue")
    plt.axhline(test_acc, label="Average Test Accuracy", linestyle='--', colour="green")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_data = dataset.ADNI(val=False, type="train", transform="train")
    val_data = dataset.ADNI(val=True, type="train", transform="val")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = modules.GFNet(in_channels=1, patch_size=14, embed_dim=384).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_EPOCHS, eta_min=LEARNING_RATE_MIN)

    top_val_acc = 0
    bad_val_count = 0

    print("~ Training start")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, True)
        val_loss, val_acc = test(model, val_loader, criterion, True)

        print(f'Epoch [{epoch}/{EPOCHS}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > top_val_acc:
            top_val_acc = val_acc
            bad_val_count = 0
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), '/Users/rorymacleod/Desktop/Uni/sem 2 24/COMP3710/Report/GFNet.pt')
        else:
            bad_val_count += 1
        if bad_val_count >= EARLY_STOP:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print("~ Training finished")

    print("~~ Testing Start")
    test_data = dataset.ADNITest()
    best_model.eval()

    preds = []
    true = []

    for data, labels in test_data:
        data, labels = data.to(device), labels.float().to(device)
        outputs = best_model(data)

        outputs = torch.sigmoid(outputs).mean().item()
        pred = 1 if outputs > 0.5 else 0

        preds.append(pred)
        true.append(labels.item())

    preds = np.array(preds)
    true = np.array(true)

    acc = accuracy_score(true, preds)
    print(f'Test Accuracy: {acc * 100:.2f}%')




# if __name__ == '__main__':
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     # load data
#     train_images, num_traing_points, val_images = dataset.load_data() 
#     test_images, num_test_points, _ = dataset.load_data(testing=True)
#     train_loader = DataLoader(train_images, batch_size=BATCH_SIZE)
#     val_loader = DataLoader(val_images, batch_size=BATCH_SIZE)
#     test_loader = DataLoader(test_images, batch_size=BATCH_SIZE)
    
#     # define and instansiate model
#     model = modules.GFNet(embed_dim=384, norm_layer=partial(nn.LayerNorm, eps=1e-6))
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.adamw(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_MAX=LR_EPOCHS, eta_min=LEARNING_RATE_MIN)
#     # define vars 
#     # prepare and complete model training
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
    
#     best_val_acc = 0
#     best_val_loss = float('inf')
#     bad_val_count = 0

#     current_epoch = 0
#     for epoch in range(EPOCHS):
#         model.train()
        
#         current_loss = 0.0
#         correct = 0
#         total = 0
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = torch.Tensor(labels).to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             current_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         train_acc = 100 * correct/total
#         avg_train_loss = current_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
#         train_accs.append(train_acc)

#         model.eval()

#         val_correct = 0
#         val_total = 0
#         val_loss = 0
#         with torch.no_grad():
#             for images, lables in val_loader:
#                 val_image = images.to(device)
#                 val_lables = torch.Tensor(labels).to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 val_total += labels.size(0)
#                 val_correct += predicted.eq(labels).sum().item()

#         val_acc = 100*val_correct/val_total
#         avg_val_loss = val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)
#         val_accs.append(val_acc)
        
#         current_epoch += 1

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             bad_val_count = 0
#             torch.save(model.state_dict(), 'best_model.pth')
#         else:
#             bad_val_count += 1
#         if bad_val_count >= EARLY_STOP:
#             print(f"Early stopping at {current_epoch}")
#             print(f"Final Train Loss: {avg_train_loss:.4f} & Acc: {train_acc:.2f}%")
#             print(f"Final Val Loss: {avg_val_loss:.4f} & Acc: {val_acc:.2f}%")
#             break
        
        # if current_epoch == 1 or current_epoch % 20 == 0:
        #     print(f'Epoch [{current_epoch}/{EPOCHS}], '
        #           f'Train Loss: {avg_train_loss:.4f}, '
        #           f'Train Acc: {train_acc:.2f}%, '
        #           f'Val Loss: {avg_val_loss:.4f}, '
        #           f'Val Acc: {val_acc:.2f}%')
#         scheduler.step(val_acc)
#     model.load_state_dict(torch.load('best_model.pth', weights_only=True))
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = torch.Tensor(labels).to(device)
#             outputs = model(images)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#     test_acc = 100* correct/total
#     print(f'Test Acc: {test_acc:.2f}%')
    
#     # Plotting training and validation loss
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, current_epoch + 1), train_losses, label='Training Loss', marker='*')
#     plt.plot(range(1, current_epoch + 1), val_losses, label='Validation Loss', marker='*')
#     plt.title('Training and Validation Loss over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("loss.png")  

#     # Plotting training and validation accuracy
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, current_epoch + 1), train_accs, label='Training Accuracy', marker='x')
#     plt.plot(range(1, current_epoch + 1), val_accs, label='Validation Accuracy', marker='x')
#     plt.title('Training and Validation Accuracy over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("acc.png") 
# def main():
#     train_dl, test_dl = dataset.create_dataloaders()

#     model = modules.GFNet().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.adamw(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#     train_accs = []
#     for epoch in range(EPOCHS):
#         train_loss, train_acc = train(model, train_dl, criterion, optimizer)
#         train_acc.append(train_acc)
#         if epoch % 10 == 0:
#             print(f"== Epoch {epoch+2}/{EPOCHS} ==")
#             print(f"Average Training ~ Loss: {train_loss:.4f} ~ Accuracy: {100*train_acc:.2f}% ")
#     torch.save(model.state_dict(), "Trained-GFNet.pth")

#     test_loss, test_acc = test(model, test_dl, criterion)
#     print(f"Average Test - Loss: {test_loss:.4f}, - Accuracy: {100*test_acc:.2f}%")

#     plot(train_accs, test_acc)
