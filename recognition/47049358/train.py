from dataset import X_train, y_train, Prostate3dDataset
from modules import ImprovedUnet
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from time import time
import numpy as np
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss

NUM_EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
LR_INITIAL = 0.985
CRITERION = DiceLoss(batch=True)
# CRITERION = DiceCELoss(batch = True, lambda_ce = 0.2) # Based on Thyroid Tumor Segmentation Report
# CRITERION = DiceFocalLoss(batch = True) # Default gamma = 2

def compute_dice_segments(predictions, ground_truths):

    criterion = DiceLoss(reduction='none', batch=True)

    num_masks = predictions.size(1)

    segment_coefs = torch.zeros(num_masks, device=ground_truths.device)

    segment_losses = criterion(predictions, ground_truths)

    for i in range(num_masks):
        
        segment_coefs[i] = 1 - segment_losses[i, : , : , : ].item()

    return segment_coefs

def train(model, X_train, y_train, criterion, num_epochs=NUM_EPOCHS, device="cuda"):

    # set up criterion, optimiser, and scheduler for learning rate. 
    optimiser = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma = LR_INITIAL)

    model.to(device)
    model.train()

    training_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_0_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_1_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_2_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_3_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_4_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_5_dice_coefs = np.zeros(NUM_EPOCHS)

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

            loss = criterion(outputs, masks) 

            segment_coefs = compute_dice_segments(outputs, masks)

            total_segment_coefs += segment_coefs

            loss.backward()

            optimiser.step()

            running_dice += 1 - loss.item()

        scheduler.step()

        for i in range(len(total_segment_coefs)):
            print(f"Epoch {epoch + 1} Segment {i} - Training Dice Coefficient: {total_segment_coefs[i] / len(train_loader)}")

        seg_0_dice_coefs[epoch] = (total_segment_coefs[0] / len(train_loader))
        seg_1_dice_coefs[epoch] = (total_segment_coefs[1] / len(train_loader))
        seg_2_dice_coefs[epoch] = (total_segment_coefs[2] / len(train_loader))
        seg_3_dice_coefs[epoch] = (total_segment_coefs[3] / len(train_loader))
        seg_4_dice_coefs[epoch] = (total_segment_coefs[4] / len(train_loader))
        seg_5_dice_coefs[epoch] = (total_segment_coefs[5] / len(train_loader))

        print(f"Epoch {epoch + 1}, Training Overall Dice Coefficient: {running_dice / len(train_loader)}")
        training_dice_coefs[epoch] = (running_dice / len(train_loader))

    return (model, training_dice_coefs, seg_0_dice_coefs, seg_1_dice_coefs,
             seg_2_dice_coefs, seg_3_dice_coefs, seg_4_dice_coefs, seg_5_dice_coefs)

# connect to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# create model. 
model = ImprovedUnet()

print("> Start Training")

start = time()

# train improved unet
trained_model, training_dice_coefs, seg0, seg1, seg2, seg3, seg4, seg5 = train(model, X_train, y_train, criterion = CRITERION,
                                                            device=device, num_epochs=NUM_EPOCHS)

end = time()

elapsed_time = end - start
print(f"> Training completed in {elapsed_time:.2f} seconds")

plt.plot(training_dice_coefs, label='Training Dice Coefficient')
plt.plot(seg0, label='Segment 0 Dice Coefficient')
plt.plot(seg1, label='Segment 1 Dice Coefficient')
plt.plot(seg2, label='Segment 2 Dice Coefficient')
plt.plot(seg3, label='Segment 3 Dice Coefficient')
plt.plot(seg4, label='Segment 4 Dice Coefficient')
plt.plot(seg5, label='Segment 5 Dice Coefficient')
plt.title(f'Dice Coefficient Over Epochs for {CRITERION}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'unet_dice_coefs_over_epochs_{CRITERION}.png')
plt.close()