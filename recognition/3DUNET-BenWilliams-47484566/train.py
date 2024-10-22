import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import modules
from dataset import MRI3DDataset
from torch import amp
import matplotlib.pyplot as plt
import numpy as np

# Example setup
#model = UNet3D(in_channels=3, out_channels=3)

#criterion = crossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]))
#optimizer = optim.Adam(model.parameters(), lr=0.001)

#epochs = 1

#device='cuda'



#inputs =  load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False)
#labels =  load_data_3D(imageNames, normImage=False, categorical=False, dtype = np.uint8, getAffines=False, orient=False, early_stop=False)


# Training loop (simplified)
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

            dice_score = modules.dice_coefficient(outputs, labels)
            epoch_dice += dice_score.item() * inputs.size(0)
            total_samples += inputs.size(0)
            if total_samples == inputs.size(0):  # Save only for the first batch
                save_example(inputs, outputs, labels, 'PatternAnalysis-2024/recognition/3DUNET-BenWilliams-47484566/images', epoch)


        epoch_loss = running_loss / total_samples
        avg_dice = epoch_dice / total_samples
        all_losses.append(epoch_loss)
        dice_scores.append(avg_dice)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {avg_dice:.4f}')
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
