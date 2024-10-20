from modules import UNet2D
from dataset import ProstateDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from utils import TEST_IMG_DIR, TEST_MASK_DIR
from utils import dice_score
from utils import N_LABELS, LABEL_NUMBER_TO_NAME


# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
])

test_set = ProstateDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transforms=transform, early_stop=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Model initialisation
model = UNet2D(in_channels=1, out_channels=6, initial_features=64, n_layers=4).to(DEVICE)
model.load_state_dict(torch.load("UNet2D_Model.pth"))

# Initialisations for Dice score tracking
total_dice_scores = torch.zeros(N_LABELS).to(DEVICE)
num_batches = 0

# Test predictions
model.eval()
with torch.no_grad():
  for images, masks in test_loader:
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    predictions = model(images)

    # Calculate the Dice score for this batch
    dice_scores = dice_score(predictions, masks)
    total_dice_scores += dice_scores
    num_batches += 1

# Calculate average Dice score for each class
average_dice_scores = total_dice_scores / num_batches
average_dice_scores = average_dice_scores.cpu()

# Display the average Dice score for each class
for i, score in enumerate(average_dice_scores):
    print(f"Average Dice Score for {LABEL_NUMBER_TO_NAME[i]}: {score.item():.4f}")
