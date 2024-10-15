import torch
from dataset import TrainPreprocessing, TestPreprocessing
from modules import GFNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# path to datasets
train_dataset_path = "../dataset/AD_NC/train"
test_dataset_path = "../dataset/AD_NC/test"

# Load train and validation data
train_data = TrainPreprocessing(train_dataset_path, batch_size=128)
train_loader, val_loader = train_data.get_train_val_loaders(val_split=0.2)

# Load test data
test_data = TestPreprocessing(test_dataset_path, batch_size=128)
test_loader = test_data.get_test_loader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_model = GFNet(
    img_size=224,
    num_classes=2,
    embed_dim=32,  # dims[0] に一致させる
    num_blocks=[1, 1, 1, 1],
    dims=[32, 64, 128, 256],  # 各ステージの次元
    drop_rate=0.05,
    drop_path_rate=0.05,
    is_training=True
)

test_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(test_model.parameters(), lr=1e-3)



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0 

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader) 
    print(f"Train Loss: {epoch_loss:.4f}")


train_one_epoch(test_model, train_loader, criterion, optimizer, device)
