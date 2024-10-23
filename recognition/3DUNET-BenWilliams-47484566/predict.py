import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRI3DDataset
from train import train_model
import modules


image_folder = 'PatternAnalysis-2024/recognition/3DUNET-BenWilliams-47484566/semantic_MRs_anon'
label_folder = 'PatternAnalysis-2024/recognition/3DUNET-BenWilliams-47484566/semantic_labels_anon'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load dataset and dataloader
dataset = MRI3DDataset(image_folder, label_folder, normImage=True)
#images = load_data_3D(image_folder, normImage=True, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, criterion, and optimizer
model = modules.UNet3D(in_channels=1, out_channels=6)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0]).to(device), ignore_index=3)
#criterion = F.cross_entropy()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dice_scores = train_model(model, dataloader, criterion, optimizer, 20000, device = device)

print(f"Final Dice scores after training: {dice_scores}")
