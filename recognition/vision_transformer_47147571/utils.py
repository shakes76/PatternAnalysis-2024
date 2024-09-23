import random
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm

def get_transform(train):
    transform_list = [
        transforms.ToTensor(),
    ]
    if train:
        transform_list.extend([
            transforms.RandomRotation(10),
            transforms.RandomApply([transforms.ElasticTransform()], p=0.5)  # Elastic Transform
        ])
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(transform_list)


def train(model, train_loader, optimizer, criterion, device="cuda", disable_tqdm=True):
    """Train the model. We assume the model output logits and train via 
    BCEWithLogitsLoss.
    disable_tqdm: Disable the progress bar
    
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, disable=disable_tqdm):
        inputs, labels = inputs.to(device), labels.float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        predicted = (outputs >= 0).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def test(model, test_loader, criterion, device="cuda", disable_tqdm=True):
    """Test the model. We assume the model output logits and train via 
    BCEWithLogitsLoss.
    use_tqdm: Show the progress bar
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, disable=disable_tqdm):
            inputs, labels = inputs.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Statistics
            predicted = (outputs >= 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    