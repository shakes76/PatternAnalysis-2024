import torch
import torchvision
from dataset import ProstateCancerDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("--Saving Checkpoint--")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("--loading checkpoint--")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_seg_dir,
        val_dir,
        val_seg_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers = 4,
        pin_memory = True,
):
    train_ds = ProstateCancerDataset(image_dir=train_dir, seg_dir=train_seg_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_ds = ProstateCancerDataset(image_dir=val_dir, seg_dir=val_seg_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader
    
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds>0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2* (preds*y).sum())/((preds+y).sum()+1e-8)
        print(f"got{num_correct}/{num_pixels} with accruacy {num_correct/num_pixels*100:.2f}")
        model.train()
