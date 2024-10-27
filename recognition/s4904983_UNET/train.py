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
from modules import UNet2D, DiceLoss
import numpy as np
import matplotlib.pyplot as plt


### TODO: Remove this
# def dice_coefficient(pred, target):
#     """ 
#     Calculate the dice coefficient for the predicted and target segmentation masks.

#     Args:
#         pred (torch.Tensor): The predicted segmentation mask.
#         target (torch.Tensor): The target segmentation mask.

#     Returns:
#         float: The dice coefficient.
#     """
#     ### TODO: Remove this
#     # new_ten = torch.add((pred[0,1,:,:] == 1)*0.2, torch.add((pred[0,2,:,:] == 1)*0.4, torch.add((pred[0,3,:,:] == 1)*0.6, torch.add((pred[0,4,:,:] == 1)*0.8, pred[0,5,:,:] == 1))))
#     # save_image(
#     #     torch.cat(
#     #         (new_ten.unsqueeze(0),
#     #         pred[0,0,:,:].unsqueeze(0),
#     #         pred[0,1,:,:].unsqueeze(0),
#     #         pred[0,2,:,:].unsqueeze(0),
#     #         pred[0,3,:,:].unsqueeze(0),
#     #         pred[0,4,:,:].unsqueeze(0),
#     #         pred[0,5,:,:].unsqueeze(0),
#     #         (target[0]/5)), 
#     #         dim=2),
#     #         os.path.join('C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/outputs/image.png')
#     #     )
#     # dice_coefficients = np.zeros(pred.shape[1])

#     # for i in range(pred.shape[0]):
#     #     for layer in range(pred.shape[1]):
#     #         intersect = ((pred[i,layer,:,:] == 1) & (target[i] == layer)).sum().item()
#     #         union = (pred[i,layer,:,:] == 1).sum().item() | (target[i] == layer).sum().item()
#     #         union += intersect # Bitwise OR avoids double counting of intersecting pixels
#     #         print(intersect, union)
#     #         if union == 0 or intersect == 0:
#     #             # Laplace smoothing
#     #             union += 1
#     #             intersect += 1/2
#     #         dice_coefficients[layer] += (2.0 * intersect) / union

#     true = torch.zeros((target.size(0), 6, target.size(1), target.size(2)))
#     # Fill the boolean tensor with the appropriate class masks
#     for i in range(6):
#         true[:, i, :, :] = (target.squeeze(1) == i)
#     # print(dice_coefficients)
#     print(pred.shape, true.shape)
#     intersect = torch.abs(torch.logical_and(pred, true)).sum()
#     union = torch.logical_or(pred, true).sum()
#     dice = 2*intersect / (union+intersect)
#     print(dice)
#     return dice


def train_unet_model():
    """ 
    Train 2D UNet model on the medical image data.

    Sets up environment, loads data, defines model, loss function and optimizer, and trains and validates the UNet model.
    """
    
    # Hyperparameters
    batch_size = 2**9
    learning_rate = 0.005
    num_epochs = 20
    rangpur = True

    # Directories for saving the output images and model
    if rangpur:
        output_dir = '/home/Student/s4904983/COMP3710/project/outputs/' # Rangpur path
        model_dir = '/home/Student/s4904983/COMP3710/project/models/' # Rangpur path
    else:
        output_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/outputs/' # Local path
        model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/models/' # Local path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    ### TODO: Remove this
    # print("Dice loss test")
    # tens1 = torch.Tensor([[[[0, 1, 1],
    #                         [1, 2, 3]]],
    #                       [[[0, 1, 1],
    #                         [1, 2, 3]]]]).long()
    # tens2 = torch.Tensor([[[
    #                         [0.99, 0.01, 0.01],
    #                         [0.01, 0.01, 0.01]
    #                     ],[
    #                         [0.1, 0.9, 0.9],
    #                         [0.9, 0.1, 0.01]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.9, 0.01]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.1, 0.9]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.1, 0.01]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.1, 0.01]
    #                     ]],
    #                     [[
    #                         [0.9, 0.1, 0.01],
    #                         [0.1, 0.1, 0.01]
    #                     ],[
    #                         [0.1, 0.9, 0.9],
    #                         [0.9, 0.1, 0.01]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.9, 0.01]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.1, 0.9]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.1, 0.01]
    #                     ],[
    #                         [0.1, 0.1, 0.01],
    #                         [0.1, 0.1, 0.01]
    #                     ]]])
    
    # print(tens1.shape, tens2.shape)

    # loss = DiceLoss()
    # test_res = loss(tens2, tens1)
    # print("Result: ", test_res)
    # assert(False)

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
    # train_dice, val_dice = [], []

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

            # print(dice_coefficient(bool_tensor, labels))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # dice = dice_coefficient(outputs, labels)
            running_loss.append(loss.item())

        train_loss.append(np.mean(running_loss))

        model.eval()
        running_loss = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(ValDataLoader):
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)

                ### TODO: Remove this
                # dice = dice_coefficient(outputs, labels)
                
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
            # val_dice.append(dice)

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