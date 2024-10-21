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


def train_unet_model():
    """ 
    Train 2D UNet model on the medical image data.

    Sets up environment, loads data, defines model, loss function and optimizer, and trains and validates the UNet model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data
    # train_images, train_segs = load_img_seg_pair('train')
    # val_images, val_segs = load_img_seg_pair('validate')

    train_images, train_segs = load_img_seg_pair('train')
    val_images, val_segs = load_img_seg_pair('validate')

    train_images = train_images[:32]
    train_segs = train_segs[:32]
    val_images = val_images[:8]
    val_segs = val_segs[:8]

    # Directory for saving the model
    model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/models/' # Local path
    # model_dir = '/home/Student/COMP3710/project/models/' # Rangpur path

    # Hyperparameters
    batch_size = 32
    learining_rate = 0.001
    num_epochs = 100
    output_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/outputs/' # Local path
    # output_dir = '/home/Student/COMP3710/project/outputs/' # Rangpur path

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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for i in range(0, len(train_images), batch_size):
            images = train_images[i:i + batch_size].to(device)
            segs = train_segs[i:i + batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(images[:,None,:,:])[:,0,:,:]
            loss = criterion(outputs, segs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, segs)

        train_loss.append(running_loss / len(train_images))
        train_dice.append(running_dice / len(train_images))

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Train Dice: {train_dice[-1]:.4f}')

        model.eval()
        running_loss = 0.0
        running_dice = 0.0

        with torch.no_grad():
            for i in range(0, len(val_images), batch_size):
                images = val_images[i:i + batch_size].to(device)
                segs = val_segs[i:i + batch_size].to(device)

                outputs = model(images[:,None,:,:])[:,0,:,:]
                loss = criterion(outputs, segs)
                
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

                # Save output images
                for j in range(batch_size):
                    save_image(images[j], os.path.join(output_dir, f'image_{epoch + 1}_{i+j}.png'))
                    save_image(segs[j], os.path.join(output_dir, f'segmentation_{epoch + 1}_{i+j}.png'))
                    save_image(outputs[j], os.path.join(output_dir, f'output_{epoch + 1}_{i+j}.png'))


        val_loss.append(running_loss / len(val_images))
        val_dice.append(running_dice / len(val_images))

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Train Dice: {train_dice[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Val Dice: {val_dice[-1]:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), model_dir + 'unet_model.pth')

    # Save the loss and dice coefficient
    np.save(output_dir + 'train_loss.npy', np.array(train_loss))
    np.save(output_dir + 'val_loss.npy', np.array(val_loss))

    # Plot the loss and dice coefficient
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir + 'loss.png')


               


def dice_coefficient(pred, target):
    """ 
    Calculate the dice coefficient for the predicted and target segmentation masks.

    Args:
        pred (torch.Tensor): The predicted segmentation mask.
        target (torch.Tensor): The target segmentation mask.

    Returns:
        float: The dice coefficient.
    """
    intersect = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    if union == 0 or intersect == 0:
        # Laplace smoothing
        union += 1e-6
        intersect += 1e-6
    return (2.0 * intersect) / union


if __name__ == '__main__':
    train_unet_model()