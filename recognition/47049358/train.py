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

NUM_EPOCHS = 10

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

NUM_EPOCHS = 10

def train(model, train_loader, valid_loader, num_epochs=100, device="cuda"):
    # Set up criterion, optimiser, and scheduler for learning rate.
    criterion = dice_coefficient
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.985)

    model.to(device)
    model.train()

    training_losses = []
    validation_losses = []

    # Initialize GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs = inputs[:, np.newaxis]
            masks = masks[:, np.newaxis]

            inputs, masks = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimiser.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast():
                outputs = model(inputs)
                loss = 1 - criterion(masks, outputs)

            # Backward pass with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item()

            # Clear cache
            torch.cuda.empty_cache()

        training_losses.append(running_loss / len(train_loader))
        scheduler.step()

        # Validation step (optional)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, masks in valid_loader:
                inputs = inputs[:, np.newaxis]
                masks = masks[:, np.newaxis]

                inputs, masks = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = 1 - criterion(masks, outputs)

                validation_loss += loss.item()

        validation_losses.append(validation_loss / len(valid_loader))
        model.train()

    return model, training_losses, validation_losses

def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    num_masks = y_true.size(-1)  # Assuming the masks are in the last dimension
    dim_coefs = torch.zeros(num_masks, device=y_true.device)  # Initialize a tensor to store the coefficients

    for i in range(num_masks):
        # Get the argmax indices along the class dimension
        ref = torch.argmax(y_pred, dim=-1)
        
        # Create binary masks where the predicted elements are set to 1 if they match the argmax indices
        y_pred_binary = (ref == i).float()
        
        y_true_f = y_true[..., i].contiguous().view(-1)  # Flatten the mask
        y_pred_f = y_pred_binary.contiguous().view(-1)  # Flatten the prediction
        
        intersection = torch.sum(y_true_f * y_pred_f)
        denominator = torch.sum(y_true_f) + torch.sum(y_pred_f)
        dim_coef = 2 * (intersection / denominator.clamp(min=epsilon))
        print(f'Mask {i}: ', dim_coef)
        dim_coefs[i] = dim_coef  # Store the coefficient in the tensor

    output = torch.divide(torch.sum(dim_coefs), num_masks)
    output.requires_grad = True
    return output

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

plt.savefig('unet_losses_over_epochs.png')