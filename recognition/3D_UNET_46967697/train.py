"""
This script trains a 3D U-Net model on the dataset. 
The model is trained using the Dice loss function and the AdamW optimizer. 

@author Damian Bellew
"""

from utils import *
from modules import *
from dataset import *
from predict import test_model

import torch
import torch.utils
from segmentation_models_pytorch.losses import DiceLoss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU")

# Data loaders
train_loader, test_loader = get_data_loaders()

# Model
model = Unet3D(IN_DIM, NUM_CLASSES, NUM_FILTERS).to(device)

# Loss and optimizer
criterion = DiceLoss(mode='multiclass', from_logits=False, smooth=SMOOTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Training the model
model.train()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        labels = labels.long().view(-1)
        labels = F.one_hot(labels, num_classes=NUM_CLASSES)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    print (f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), MODEL_PATH)

# Test the model
test_model(device, model, test_loader, criterion)
