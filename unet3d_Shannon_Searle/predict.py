# predict.py
import torch
from tqdm import tqdm
import torch.nn.functional as F

# Function to calculate Dice coefficient for multi-class segmentation
def dice_coefficient(predicted, target, smooth=1e-6):
    num_classes = predicted.shape[1]  # Assuming [batch_size, num_classes, depth, height, width]
    predicted = torch.argmax(predicted, dim=1)  # Convert predictions to class indices
    predicted_one_hot = F.one_hot(predicted, num_classes).permute(0, 4, 1, 2, 3).float()

    target = torch.argmax(target, dim=1)  # Convert target to class indices
    target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

    # Flatten the tensors
    predicted_flat = predicted_one_hot.reshape(-1, num_classes)
    target_flat = target_one_hot.reshape(-1, num_classes)

    intersection = (predicted_flat * target_flat).sum(dim=0)
    dice_score = (2. * intersection + smooth) / (predicted_flat.sum(dim=0) + target_flat.sum(dim=0) + smooth)

    return dice_score.mean().item()  # Return mean Dice coefficient across all classes

# Function to evaluate Dice coefficient on a test dataset. 
# returns avg dice coefficient and true/false if all labels have a minimum Dice similarity coefficient of 0.7 
def evaluate_dice_on_test_set(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_dice = 0.0
    num_samples = len(test_loader)
    min_coeff = True
    with torch.no_grad():  # Disable gradient computation for inference
        for images, labels in tqdm(test_loader, disable = True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Calculate Dice coefficient for the batch
            dice_score = dice_coefficient(outputs, labels)
            total_dice += dice_score
            if dice_score < 0.7:
                min_coeff = False
    # Return the average Dice coefficient over the test set
    avg_dice = total_dice / num_samples
    return avg_dice, min_coeff