import torch
from torch.utils.data import DataLoader
from modules import UNet, dice_coefficient
from dataset import NiftiDataset, load_nifti_images_from_folder

folder_path = r"D:\HuaweiMoveData\Users\HUAWEI\Desktop\keras_slice"
image_names = load_nifti_images_from_folder(folder_path)
dataset = NiftiDataset(image_names, norm_image=True, categorical=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('model.pth'))  # Load your trained model

model.eval()
with torch.no_grad():
    for images in dataloader:
        images = images.to(device)
        predictions = model(images)
        dice_score = dice_coefficient(labels, predictions)
        print(f'Dice Similarity Coefficient: {dice_score.item():.4f}')
