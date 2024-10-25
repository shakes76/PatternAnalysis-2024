""" 
File: train.py
Author: Øystein Kvandal
Description: Contains the training loop for the 2D UNET model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import MRIDataLoader, MRIDataset
from modules import UNet2D
import numpy as np
import matplotlib.pyplot as plt


def dice_coefficient(pred, target):
    """ 
    Calculate the dice coefficient for the predicted and target segmentation masks.

    Args:
        pred (torch.Tensor): The predicted segmentation mask.
        target (torch.Tensor): The target segmentation mask.

    Returns:
        float: The dice coefficient.
    """
    target = target[:,None,:,:]
    num_classes = pred.shape[1]
    true = torch.eye(num_classes)[target.squeeze(1)]
    true = true.permute(0, 3, 1, 2).float()
    probabilities = F.softmax(pred, dim=1)
    dims = (0,) + tuple(range(2, true.ndimension()))

    intersect = torch.sum(probabilities * true, dims).sum()
    total_sum = torch.sum(probabilities + true, dims).sum()
    if total_sum == 0 or intersect == 0:
        # Laplace smoothing
        total_sum += 1e-6
        intersect += 1e-6
    return (2.0 * intersect) / total_sum


def train_unet_model():
    """ 
    Train 2D UNet model on the medical image data.

    Sets up environment, loads data, defines model, loss function and optimizer, and trains and validates the UNet model.
    """
    
    # Hyperparameters
    batch_size = 2**6
    learining_rate = 0.005
    num_epochs = 100
    # output_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/outputs/' # Local path
    output_dir = '/home/Student/s4904983/COMP3710/project/outputs/' # Rangpur path

    # Directory for saving the model
    # model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/models/' # Local path
    model_dir = '/home/Student/s4904983/COMP3710/project/models/' # Rangpur path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data
    print("Initializing datasets and dataloaders")
    TrainDataLoader = MRIDataLoader("train", batch_size=batch_size)
    print("Training data initialized")
    ValDataLoader = MRIDataLoader("validate", batch_size=batch_size)
    print("Validation data initialized")


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the model
    model = UNet2D().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learining_rate)

    train_loss, val_loss = [], []
    train_dice, val_dice = [], []

    print('\nStarting training:\n')

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_dice = 0.0, 0.0

        # tqdm(total=num_epochs, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
        for i, (images, labels) in enumerate(tqdm(TrainDataLoader, desc=f'Epoch {epoch + 1:>3}/{num_epochs:<3}', unit='batch')):
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            dice = dice_coefficient(outputs, labels[:,0,:,:])
            loss = criterion(outputs, labels[:,0,:,:])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice

        train_loss.append(running_loss / len(TrainDataLoader.dataset))
        train_dice.append(running_dice / len(TrainDataLoader.dataset))

        model.eval()
        running_loss, running_dice = 0.0, 0.0


        with torch.no_grad():
            for i, (images, labels) in enumerate(ValDataLoader):
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels[:,0,:,:])
                
                # if (epoch + 1 ) % 10 == 0:
                #     for i in range(256):
                #         string = ""
                #         l = ['{0:.1f}'.format(float(outputs[0][i][j].cpu().numpy())) for j in range(128)]
                #         string += " ".join(l)
                #         print(string)
                #     plt.figure(1)
                #     plt.imshow(outputs[0].cpu().numpy())
                #     plt.figure(2)
                #     plt.imshow(labels[0].cpu().numpy())
                #     plt.figure(3)
                #     plt.imshow(images[0].cpu().numpy())
                #     plt.show()


                running_loss += loss.item()
                running_dice += dice_coefficient(outputs, labels[:,0,:,:])
                if (epoch+1) % 10 == 0 and i % 5 == 0:
                    # Save output images
                    for j in range(0, batch_size, int(batch_size/2+0.5)):
                        image_index = i*batch_size + j
                        save_image(
                            torch.cat(
                                (images[j], 
                                outputs[j,0,:,:].unsqueeze(0),
                                outputs[j,1,:,:].unsqueeze(0),
                                outputs[j,2,:,:].unsqueeze(0),
                                outputs[j,3,:,:].unsqueeze(0),
                                outputs[j,4,:,:].unsqueeze(0),
                                outputs[j,5,:,:].unsqueeze(0),
                                (labels[j]/4)), 
                                dim=2),
                                os.path.join(output_dir, f'e{epoch + 1}_im{image_index}.png')
                            )
                        # save_image(images[j], os.path.join(output_dir, f'e{epoch + 1}_{i+j}_image.png'))
                        # save_image((labels[j]/4), os.path.join(output_dir, f'e{epoch + 1}_{i+j}_segmentation.png'))
                        # save_image((torch.round(torch.mul(outputs[j], 4))/4), os.path.join(output_dir, f'e{epoch + 1}_{i+j}_output.png'))


            val_loss.append(running_loss / len(ValDataLoader.dataset))
            val_dice.append(running_dice / len(ValDataLoader.dataset))

            print(f'{" ":>19}{"|":<8}Train Loss: {train_loss[-1]:.4f} {"|":^15} Val Loss: {val_loss[-1]:.4f} {"|":^15} Train Dice: {train_dice[-1]:.4f} {"|":^15} Val Dice: {val_dice[-1]:.4f}')
    
        if (epoch + 1) % 20 == 0:
            # Save the model
            torch.save(model.state_dict(), model_dir + 'unet_model_ep' + str(epoch + 1) + '.pth')

            # Save the loss and dice coefficient
            np.save(output_dir + 'train_loss_ep' + str(epoch + 1) + '.npy', np.array(train_loss))
            np.save(output_dir + 'val_loss_ep' + str(epoch + 1) + '.npy', np.array(val_loss))

            # Plot the loss and dice coefficient
            plt.figure()
            plt.plot(train_loss, label='Train Loss')
            plt.plot(val_loss, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(output_dir + 'loss_ep' + str(epoch + 1) + '.png')


if __name__ == '__main__':
    train_unet_model()