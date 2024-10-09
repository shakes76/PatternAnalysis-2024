"""
Contains the code for training, validating, testing, and saving the Unet
"""
from modules import Improved3DUnet
from metrics import get_loss_function
from dataset import ProstateDataset, TOP_LEVEL_DATA_DIR, SEMANTIC_LABELS, SEMANTIC_MRS

import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import time

# hyperparameters
BATCH_SIZE = 2
EPOCHS = 10
NUM_CLASSES = 6 # as per powerpoint slides
INPUT_CHANNELS = 1 # greyscale
NUM_LOADED = 10 # set to None to load all
SHUFFLE = False
WORKERS = 3

# taken from the paper on the improved unet
INITIAL_LR = 5e-4
WEIGHT_DECAY = 1e-5
DECAY_FACTOR = 0.985

# setup the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print(f"Warning! CUDA not found, using CPU")
else:
    print(f"Running on device {device}")
torch.autograd.set_detect_anomaly(True) # help detect problems

# setup data
dataset = ProstateDataset(TOP_LEVEL_DATA_DIR + SEMANTIC_MRS, TOP_LEVEL_DATA_DIR + SEMANTIC_LABELS,
                          INPUT_CHANNELS, NUM_CLASSES, num_load=NUM_LOADED)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

# setup model
model = Improved3DUnet(INPUT_CHANNELS, NUM_CLASSES)
# optimiser taken from the paper on the improved unet
optimiser = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda epoch: DECAY_FACTOR ** epoch)
# will use dice loss as per the task requirements
criterion = get_loss_function()

model = model.to(device)

# ================== training procedure
model.train()
print(f"Training...")
start = time.time()

for epoch in range(EPOCHS):
    for step, (image, mask) in enumerate(data_loader):
        image = image.to(device)
        mask = mask.to(device)

        outputs = model(image)
        loss = criterion(outputs, mask)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    scheduler.step()

end = time.time()
elapsed = end - start

print(f"Training took {str(elapsed)} secs or {str(elapsed/60)} mins in total")