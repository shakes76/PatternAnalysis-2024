"""
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

from dataset import train_loader, validation_loader
from modules import ImprovedUnet
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

NUM_EPOCHS = 1

def train(model, train_loader, valid_loader, num_epochs=100, device="cuda"):
    # set up criterion, optimiser, and scheduler for learning rate. 

    criterion = dice_coefficient
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.985)


    model.to(device)
    model.train()

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            
            inputs = inputs[np.newaxis, :]
            masks = masks[np.newaxis, :]

            inputs, masks = inputs.to(device), masks.to(device)

            optimiser.zero_grad()
            outputs = model(inputs)

            print(outputs)

            # we want to maximise the dice coefficient
            # loss is then 1 - dice coefficient 
            loss = 1 - criterion(masks, outputs) 
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

            print(running_loss)

            break

        scheduler.step()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0

        # get validation losses. 
        with torch.no_grad():
            for val_inputs, val_masks in valid_loader:

                val_inputs = val_inputs[np.newaxis, :]
                val_masks = val_masks[np.newaxis, :]

                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)
                val_outputs = model(val_inputs)
                val_loss += 1 - criterion(val_outputs, val_masks).item()

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(valid_loader)}")

        training_losses.append(running_loss / len(train_loader))
        validation_losses.append(val_loss / len(valid_loader))

    return model, training_losses, validation_losses

def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    smooth = 1e-8
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (2. * intersection + torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return dice

# connect to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# create model. 
model = ImprovedUnet()

# train improved unet
trained_model, training_losses, validation_losses = train(model, train_loader, validation_loader, 
                                                            device=device, num_epochs=NUM_EPOCHS)

plt.figure(figsize=(10,5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Losses over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()