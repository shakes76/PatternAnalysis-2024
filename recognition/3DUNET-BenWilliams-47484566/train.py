import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import modules
from dataset import MRI3DDataset


# Example setup
#model = UNet3D(in_channels=3, out_channels=3)

#criterion = crossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]))
#optimizer = optim.Adam(model.parameters(), lr=0.001)

#epochs = 1

#device='cuda'



#inputs =  load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False)
#labels =  load_data_3D(imageNames, normImage=False, categorical=False, dtype = np.uint8, getAffines=False, orient=False, early_stop=False)


# Training loop (simplified)
def train_model(model, criterion, optimizer):
    model = model.to(device)
    model.train()

    dice_scores = []
    for epoch in range(epochs):
        loss = 0.0
        epoch_dice = 0.0

        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs
            loss = criterion(outputs, labels))

            loss.backward()
            optimizer.step()

            loss += loss.item() * inputs.size(0)    

        dice_score = dice_coefficient(outputs, labels)
        epoch_dice += dice_score.item() * inputs.size(0)
        
        dice_scores.append(avg_dice)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {avg_dice:.4f}')
    return dice_scores
