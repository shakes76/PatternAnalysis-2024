import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet, dice_coefficient
from dataset import NiftiDataset, load_nifti_images_from_folder

folder_path = r"D:\HuaweiMoveData\Users\HUAWEI\Desktop\keras_slice"
image_names = load_nifti_images_from_folder(folder_path)
dataset = NiftiDataset(image_names, norm_image=True, categorical=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for images in dataloader:
        images = images.to(device)
        labels = ...  # Replace with appropriate labels

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
