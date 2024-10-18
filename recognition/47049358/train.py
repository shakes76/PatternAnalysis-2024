"""
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

from dataset import X_train, y_train, Prostate3dDataset
from modules import ImprovedUnet
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from time import time
import math


NUM_EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
LR_INITIAL = 0.985
LOSS_IDX = 4

class BaseDice(nn.Module):
    def __init__(self, epsilon = 1e-7):
        super(BaseDice, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        raise NotImplementedError("Sublasses should implement this method.")

class ExponentialWeightedLoss(BaseDice):
    def __init__(self, epsilon=1e-7):
        super().__init__(epsilon)

    def __str__(self):
        return 'ExponentialWeightedLoss'

    def forward(self, y_pred, y_true):

        num_masks = y_true.size(-1)
        segment_coefs = torch.zeros(num_masks, device=y_true.device)
        for i in range(num_masks):
            ground_truth_seg = y_true[:, :, :, :, i]
            pred_seg = y_pred[:, :, :, :, i]

            d_coef = (2 * torch.sum(torch.mul(ground_truth_seg, pred_seg))) / (torch.sum(ground_truth_seg + pred_seg) + self.epsilon)
            segment_coefs[i] = d_coef

        weighted, _ = torch.sort(segment_coefs)

        for i in range(num_masks):
            weighted[i] = segment_coefs[i] / (math.e ** i)
        
        d_coef = (1 / num_masks) * torch.sum(segment_coefs)
        loss = 1 - (1 / num_masks) * torch.sum(weighted)
        return loss, segment_coefs, d_coef
    
class ArithmeticWeightedLoss(BaseDice):
    def __init__(self, epsilon=1e-7):
        super().__init__(epsilon)

    def __str__(self):
        return 'ArithmeticWeightedLoss'

    def forward(self, y_pred, y_true):

        num_masks = y_true.size(-1)
        segment_coefs = torch.zeros(num_masks, device=y_true.device)
        for i in range(num_masks):
            ground_truth_seg = y_true[:, :, :, :, i]
            pred_seg = y_pred[:, :, :, :, i]

            d_coef = (2 * torch.sum(torch.mul(ground_truth_seg, pred_seg))) / (torch.sum(ground_truth_seg + pred_seg) + self.epsilon)
            segment_coefs[i] = d_coef

        weighted, _ = torch.sort(segment_coefs)

        for i in range(num_masks):
            weighted[i] = segment_coefs[i] / (i + 1)
        
        d_coef = (1 / num_masks) * torch.sum(segment_coefs)
        loss = 1 - (1 / num_masks) * torch.sum(weighted)
        return loss, segment_coefs, d_coef
    
class PaperLoss(BaseDice):
    def __init__(self, epsilon=1e-7):
        super().__init__(epsilon)

    def __str__(self):
        return 'PaperLoss'

    def forward(self, y_pred, y_true):

        num_masks = y_true.size(-1)
        segment_coefs = torch.zeros(num_masks, device=y_true.device)
        for i in range(num_masks):
            ground_truth_seg = y_true[:, :, :, :, i]
            pred_seg = y_pred[:, :, :, :, i]

            d_coef = (2 * torch.sum(torch.mul(ground_truth_seg, pred_seg))) / (torch.sum(ground_truth_seg + pred_seg) + self.epsilon)
            segment_coefs[i] = d_coef

        loss = (- 1 / num_masks) * torch.sum(segment_coefs)
        d_coef = (1 / num_masks) * torch.sum(segment_coefs)
        return loss, segment_coefs, d_coef
    
class AlternativeLoss(BaseDice):
    def __init__(self, epsilon=1e-7):
        super().__init__(epsilon)

    def __str__(self):
        return 'AlternativeLoss'

    def forward(self, y_pred, y_true):

        num_masks = y_true.size(-1)
        segment_coefs = torch.zeros(num_masks, device=y_true.device)
        for i in range(num_masks):
            ground_truth_seg = y_true[:, :, :, :, i]
            pred_seg = y_pred[:, :, :, :, i]

            d_coef = (2 * torch.sum(torch.mul(ground_truth_seg, pred_seg))) / (torch.sum(ground_truth_seg + pred_seg) + self.epsilon)
            segment_coefs[i] = d_coef

        d_coef = (1 / num_masks) * torch.sum(segment_coefs)
        loss = 1 - (1 / num_masks) * torch.sum(segment_coefs)
        return loss, segment_coefs, d_coef
    
class PaperLossPlus(BaseDice):
    def __init__(self, epsilon=1e-7):
        super().__init__(epsilon)

    def __str__(self):
        return 'PaperLossPlus'

    def forward(self, y_pred, y_true):

        num_masks = y_true.size(-1)
        bce = nn.BCELoss()
        segment_coefs = torch.zeros(num_masks, device=y_true.device)

        for i in range(num_masks):
            ground_truth_seg = y_true[:, :, :, :, i]
            pred_seg = y_pred[:, :, :, :, i]

            d_coef = (2 * torch.sum(torch.mul(ground_truth_seg, pred_seg))) / (torch.sum(ground_truth_seg + pred_seg) + self.epsilon)
            segment_coefs[i] = d_coef

        d_coef = (1 / num_masks) * torch.sum(segment_coefs)
        loss = (- 1 / num_masks) * torch.sum(segment_coefs) + bce(y_pred, y_true)
        return loss, segment_coefs, d_coef

def train(model, X_train, y_train, loss, num_epochs=NUM_EPOCHS, device="cuda"):

    # set up criterion, optimiser, and scheduler for learning rate. 
    criterion = loss
    optimiser = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma = LR_INITIAL)

    model.to(device)
    model.train()

    training_dice_coefs = []

    train_set = Prostate3dDataset(X_train, y_train)
    train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE)

    for epoch in range(num_epochs):
        running_dice = 0.0
        total_segment_coefs = torch.zeros(y_train.shape[-1], device=device)
        for inputs, masks in train_loader:
            masks = masks.float()
            inputs, masks = inputs.to(device), masks.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)

            # the weighted value is only used for updating gradients!
            loss, segment_coefs, d_coef = criterion(y_pred = outputs, y_true = masks) 

            total_segment_coefs += segment_coefs

            loss.backward()

            optimiser.step()

            running_dice += d_coef.item()

        scheduler.step()

        for i in range(len(total_segment_coefs)):
            print(f"Epoch {epoch + 1} Segment {i} - Training Dice Coefficient: {total_segment_coefs[i] / len(train_loader)}")

        print(f"Epoch {epoch + 1}, Training Overall Dice Coefficient: {running_dice / len(train_loader)}")
        training_dice_coefs.append(running_dice / len(train_loader))

    return model, training_dice_coefs

# connect to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# create model. 
model = ImprovedUnet()

# Importing Dataloader breaks the implementation. Hence they are loaded below instead:

loss_map = {0 : PaperLoss(), 1 : AlternativeLoss(), 2 : ExponentialWeightedLoss(), 3 : ArithmeticWeightedLoss(), 4 : PaperLossPlus()}

loss = loss_map.get(LOSS_IDX)

print("> Start Training")

start = time()

# train improved unet
trained_model, training_dice_coefs = train(model, X_train, y_train, loss = loss,
                                                            device=device, num_epochs=NUM_EPOCHS)

end = time()

elapsed_time = end - start
print(f"> Training completed in {elapsed_time:.2f} seconds")

plt.figure(figsize=(10,5))
plt.plot(training_dice_coefs, label='Training Dice Coefficient')
plt.title('Dice Coefficient Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'unet_dice_coefs_over_epochs_{str(loss)}.png')
plt.close()