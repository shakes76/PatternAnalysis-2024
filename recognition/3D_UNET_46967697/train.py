import torch.utils
from utils import *
from modules import *
from dataset import *

import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU")

# Data
data = Prostate3DDataset(SEMANTIC_MRS_PATH, SEMANTIC_LABELS_PATH)
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_data, test_data = torch.utils.data.random_split(data, [TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT], generator=generator)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data)
test_loader = torch.utils.data.DataLoader(dataset=test_data)

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



