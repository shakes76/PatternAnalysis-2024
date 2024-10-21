"""
train.py
"""
from const import DATASET_PATH, NET_OUTPUT_TARGET
from dataset import mri_split, MriData3D
from modules import FullUNet3D
from torch.utils.data import DataLoader
import torch
from torch import nn, optim

# CHECK CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("NO CUDA AVAILABLE. CPU IN USE")
print(device)

# Sum Training Options
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 1
NUM_CLASSES = 6

files_train, files_test, files_validate = mri_split(data_path=DATASET_PATH,proportions=[0.7, 0.2, 0.1])
# print(len(files_train + files_test + files_validate))

data_train = MriData3D(data_path=DATASET_PATH,target_data=files_train)
data_test = MriData3D(data_path=DATASET_PATH,target_data=files_test)
data_validate = MriData3D(data_path=DATASET_PATH,target_data=files_validate)

train_dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)


model = FullUNet3D(input_width=1,out_width=NUM_CLASSES).to(device=device)
model = model.to(device)

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Start Training")
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_dataloader):
        images = torch.tensor(images,device=device)
        labels = torch.tensor(labels,device=device)
        b_size = images.size(0)

        # --- Train ---
        model.train()
        # Forward pass
        out:torch.Tensor = model(images)

        # Compute Error, backpropagate, optimize
        # out = out.to(device='cpu')
        calc_loss = loss(out.to(torch.float),labels.squeeze(dim=1).to(torch.long))
        calc_loss.backward()
        optimizer.step()

        # Training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, NUM_EPOCHS, i, len(train_dataloader),
                     calc_loss.item()), flush=True)

        # Save loss
        # losses.append(calc_loss.item())


# Save model
torch.save(model, NET_OUTPUT_TARGET)
