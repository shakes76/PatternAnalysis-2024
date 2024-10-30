import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modules import UNet
from dataset import ProstateDataset
from torch.utils.data import DataLoader


batch_size = 32
N_epochs = 3
n_workers = 2
pin = True
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model = False
img_height = 128
img_width = 64
learning_rate = 0.0005

# Data directories
train_image_dir = 'keras_slices_data/keras_slices_train'
train_mask_dir = 'keras_slices_data/keras_slices_seg_train'
val_image_dir = 'keras_slices_data/keras_slices_validate'
val_mask_dir = 'keras_slices_data/keras_slices_seg_validate'

def train_fn(loader,model,optimizer,loss_fn,scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=device)
        targets = data.float().unsqueeze(1).to(device=device)

        with torch.amp.autocast(device_type=device):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_trainsform = A.Compose(
       [ A.Resize(height=img_height,width=img_width),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2()]

    )

    train_dataset = ProstateDataset(train_image_dir, train_mask_dir, norm_image=True, transform=train_trainsform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = ProstateDataset(val_image_dir, val_mask_dir, norm_image=True,transform=train_trainsform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    model = UNet(in_channels=1,out_channels=6).to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(N_epochs):
        train_fn(train_loader,model,optimizer,loss_fn,scaler)


def dice_score(loader, model, num_classes=6):
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # Get model predictions
            preds = model(x)
            preds = preds.argmax(dim=1)  # Predicted class labels per pixel

            # Initialize dice score for the batch
            dice_score_batch = 0.0

            # Compute Dice score for each class
            for cls in range(num_classes):
                pred_cls = (preds == cls).float()
                true_cls = (y == cls).float()

                intersection = (pred_cls * true_cls).sum()
                union = pred_cls.sum() + true_cls.sum()

                dice_cls = (2.0 * intersection + 1e-8) / (union + 1e-8)
                dice_score_batch += dice_cls

            # Average Dice score over all classes for the batch
            dice_score_batch /= num_classes
            total_dice += dice_score_batch

    # Average Dice score over all batches
    dice = total_dice / len(loader)
    print(f"Dice Score: {dice.item():.4f}")


def save_img(loader,model,folder="images", device="cuda"):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)# some how get preds for multiclass cant use sigmoid
            preds = preds.float()
            torchvision.utils.save_image(preds, f"{folder}/prediction_{idx}.png")

if __name__ == '__main__':
    main()
