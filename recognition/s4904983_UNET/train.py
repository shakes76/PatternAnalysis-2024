""" 
File: train.py
Author: Øystein Kvandal
Description: Contains the training loop for the 2D UNET model.
"""

import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from modules import UNet2D, DiceLoss
import dataset as ds
from dataset import MRIDataLoader, MRIDataset
import numpy as np
import matplotlib.pyplot as plt


def train_unet_model():
    """ 
    Train 2D UNet model on the medical image data.

    Sets up environment, loads data, defines model, loss function and optimizer, and trains and validates the UNet model.
    """
    
    # Hyperparameters
    batch_size = 2**9
    learning_rate = 0.005
    num_epochs = 20

    # Directories for saving the output images and model
    if ds.IS_RANGPUR_ENV:
        output_dir = '/home/Student/s4904983/COMP3710/project/outputs/' # Rangpur path
        model_dir = '/home/Student/s4904983/COMP3710/project/models/' # Rangpur path
    else:
        output_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/outputs/' # Local path
        model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/models/' # Local path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Load the data
    print("Initializing datasets and dataloaders")
    TrainDataLoader = MRIDataLoader("train", batch_size=batch_size, shuffle=True)
    print("Training data initialized")
    ValDataLoader = MRIDataLoader("validate", batch_size=batch_size, shuffle=True)
    print("Validation data initialized")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the model
    model = UNet2D().to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss, val_loss = [], []

    print('\nStarting training:\n')

    for epoch in range(num_epochs):
        model.train()
        running_loss = []

        for i, (images, labels) in enumerate(TrainDataLoader):
            images, labels = images.to(device), labels.to(device).long()
            
            ### TODO: Remove this
            # bool_tensor = torch.zeros((labels.size(0), 6, labels.size(2), labels.size(3)))
            # # Fill the boolean tensor with the appropriate class masks
            # for cls in range(6):
            #     bool_tensor[:, cls, :, :] = (labels.squeeze(1) == cls)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())

        train_loss.append(np.mean(running_loss))

        model.eval()
        running_loss = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(ValDataLoader):
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss.append(loss.item())
                if (epoch+1) % 5 == 0 and i % 5 == 0:
                    # Save output images
                    for j in range(0, batch_size, int(batch_size/2-0.5)):
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
                        # save_image((torch.round(torch.(outputs[j], 4))/4), os.path.join(output_dir, f'e{epoch + 1}_{i+j}_output.png'))


            val_loss.append(np.mean(running_loss))

            print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f} | Val Loss: {val_loss[-1]:.4f}')
    
        if (epoch + 1) % 5 == 0:
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