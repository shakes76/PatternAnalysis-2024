import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import modules
from dataset import MRI3DDataset
from torch import amp
import matplotlib.pyplot as plt
import numpy as np
import os

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs, device='cuda'):
    model = model.to(device)
    model.train()
    scaler = amp.GradScaler(device=device)
    all_losses = []
    dice_scores = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_dice = 0.0
        total_samples = 0
        dice_scores_epoch = []
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):  # Enable mixed precision
                outputs = model(inputs)
                labels = torch.squeeze(labels, dim=1)
                loss = criterion(outputs, labels)
            torch.cuda.empty_cache()

            scaler.scale(loss).backward()  # Scale the loss and call backward
            scaler.step(optimizer)  # Update weights
            scaler.update()

            running_loss += loss.item() * inputs.size(0)    

            #dice_score = modules.dice_coefficient(outputs, labels)
            #epoch_dice += dice_score.item() * inputs.size(0)
            total_samples += inputs.size(0)

            dice_scores = modules.dice_coefficient_per_label(outputs, labels, num_classes=6)  # Change num_classes if needed
            dice_scores_epoch.append(dice_scores)
        
        epoch_loss = running_loss / total_samples
        all_losses.append(epoch_loss)
        dice_scores.append(dice_scores_epoch)
        #torch.save(model.state_dict(), f'model_weights_epoch_{epoch + 1}.pth')
        avg_dice = {label: sum(dice[label] for dice in dice_scores_overall) / len(dice_scores_overall) for label in range(6)}
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {avg_dice:.4f}')
    torch.save(model.state_dict(), f'model_weights_epoch_{epoch + 1}.pth')
    plot_loss(all_losses, 'PatternAnalysis-2024/recognition/3DUNET-BenWilliams-47484566/images')
    return dice_scores





def save_example(inputs, outputs, labels, output_dir, epoch):
    # Convert tensors to NumPy arrays and take the first example from the batch
    input_image = inputs[0].cpu().detach().numpy()
    output_image = outputs[0].cpu().detach().numpy()
    label_image = labels[0].cpu().detach().numpy()

    # Save images (you may want to adjust the channel and value ranges)
    np.save(os.path.join(output_dir, f'input_epoch_{epoch}.npy'), input_image)
    np.save(os.path.join(output_dir, f'output_epoch_{epoch}.npy'), output_image)
    np.save(os.path.join(output_dir, f'label_epoch_{epoch}.npy'), label_image)

def plot_loss(losses, output_dir):
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
