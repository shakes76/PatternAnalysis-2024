""" 
File: train.py
Author: Øystein Kvandal
Description: Contains the training loop for the 2D UNET model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import load_img_seg_pair
from modules import UNet2D
import numpy as np
import matplotlib.pyplot as plt


def dice_coefficient(pred, target, margin=0.0):
    """ 
    Calculate the dice coefficient for the predicted and target segmentation masks.

    Args:
        pred (torch.Tensor): The predicted segmentation mask.
        target (torch.Tensor): The target segmentation mask.

    Returns:
        float: The dice coefficient.
    """
    intersect = ((pred-target).abs() <= margin).sum().item()
    total_sum = (pred > margin).sum().item() + (target > margin).sum().item()
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data
    print("Loading data")
    train_images, train_segs = load_img_seg_pair('train')
    print("Training data loaded")
    val_images, val_segs = load_img_seg_pair('validate')
    print("Validation data loaded")

    train_images = train_images
    train_segs = train_segs
    val_images = val_images
    val_segs = val_segs

    # Directory for saving the model
    # model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/models/' # Local path
    model_dir = '/home/Student/COMP3710/project/models/' # Rangpur path

    # Hyperparameters
    batch_size = 1
    learining_rate = 0.001
    num_epochs = 60
    # output_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/outputs/' # Local path
    output_dir = '/home/Student/COMP3710/project/outputs/' # Rangpur path

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
    discrete_val_loss = []
    discrete_val_dice = []

    print('\nStarting training:\n')

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_dice = 0.0, 0.0

        with tqdm(total=8, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for i in range(0, len(train_images), batch_size):
                images = train_images[i:i + batch_size].to(device)
                segs = train_segs[i:i + batch_size].to(device)

                optimizer.zero_grad()
                outputs = model(images[:,None,:,:])[:,0,:,:]
                loss = criterion(outputs, segs) / outputs.numel()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                running_loss += loss.item()
                running_dice += dice_coefficient(outputs, segs)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            train_loss.append(running_loss / len(train_images))
            train_dice.append(running_dice / len(train_images))

            model.eval()
            running_loss, running_dice = 0.0, 0.0
            disc_run_loss, disc_run_dice = 0.0, 0.0


        with torch.no_grad():
            for i in range(0, len(val_images), batch_size):
                images = val_images[i:i + batch_size].to(device)
                segs = val_segs[i:i + batch_size].to(device)

                outputs = model(images[:,None,:,:])[:,0,:,:]
                loss = criterion(outputs, segs) / outputs.numel()
                
                # if (epoch + 1 ) % 10 == 0:
                #     for i in range(256):
                #         string = ""
                #         l = ['{0:.1f}'.format(float(outputs[0][i][j].cpu().numpy())) for j in range(128)]
                #         string += " ".join(l)
                #         print(string)
                #     plt.figure(1)
                #     plt.imshow(outputs[0].cpu().numpy())
                #     plt.figure(2)
                #     plt.imshow(segs[0].cpu().numpy())
                #     plt.figure(3)
                #     plt.imshow(images[0].cpu().numpy())
                #     plt.show()


                running_loss += loss.item()
                running_dice += dice_coefficient(outputs, segs)
                disc_run_loss += criterion(torch.round(torch.mul(outputs, 4)), segs) / outputs.numel()
                disc_run_dice += dice_coefficient(torch.round(torch.mul(outputs, 4)), segs)

                # Save output images
                for j in range(batch_size):
                    save_image(
                        torch.cat(
                            (images[j].unsqueeze(0), 
                            outputs[j].unsqueeze(0), 
                            (torch.round(torch.mul(outputs[j], 4))/4).unsqueeze(0), 
                            (segs[j]/4).unsqueeze(0)), 
                            dim=2),
                            os.path.join(output_dir, f'e{epoch + 1}_im{i+j}.png')
                        )
                    # save_image(images[j], os.path.join(output_dir, f'e{epoch + 1}_{i+j}_image.png'))
                    # save_image((segs[j]/4), os.path.join(output_dir, f'e{epoch + 1}_{i+j}_segmentation.png'))
                    # save_image((torch.round(torch.mul(outputs[j], 4))/4), os.path.join(output_dir, f'e{epoch + 1}_{i+j}_output.png'))


        val_loss.append(running_loss / len(val_images))
        val_dice.append(running_dice / len(val_images))
        discrete_val_loss.append(disc_run_loss / len(val_images))
        discrete_val_dice.append(disc_run_dice / len(val_images))

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Discrete Val Loss: {discrete_val_loss[-1]:.4f}, Train Dice: {train_dice[-1]:.4f}, Val Dice: {val_dice[-1]:.4f}, Discrete Val Dice: {discrete_val_dice[-1]:.4f}')
    
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