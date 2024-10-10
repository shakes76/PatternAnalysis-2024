import torch.utils
from utils import *
from modules import *
from dataset import *

import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU")

# Data loaders
train_loader, test_loader = get_data_loaders()

# Model
model = Unet3D(IN_DIM, NUM_CLASSES, NUM_FILTERS).to(device)

# Loss and optimizer
criterion = DiceLoss()
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
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    print (f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), '3d_unet_model.pth')


# Test the model
model.eval()
test_loss = 0.0
dice_score = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Prediction
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        # Compute dice score
        #preds = torch.sigmoid(outputs)  
        #preds = (preds > 0.5).float()  

        #intersection = (preds * labels).sum() 
        #dice = (2. * intersection + 1e-6) / (preds.sum() + labels.sum() + 1e-6)
        #dice_score += dice.item()

# Compute the average test loss and Dice score
avg_test_loss = test_loss / len(test_loader)
#avg_dice_score = dice_score / len(test_loader)

print(f'Average Test Loss: {avg_test_loss}')
#print(f'Average Dice Score: {avg_dice_score}')
