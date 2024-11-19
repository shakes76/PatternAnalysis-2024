import torch
from modules import UNET
from dataset import ProstateCancerDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import visualize_predictions, check_accuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load the trained model
def load_model(model_path, model):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


# Function to run predictions
def predict():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = "my_checkpoint.pth.tar"

    # Load model
    model = UNET(in_channels=1, out_channels=5).to(DEVICE)
    model = load_model(MODEL_PATH, model)
    model.eval()

    # Load the test dataset
    TEST_IMG_DIR = "C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_test"
    TEST_SEG_DIR = "C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_test"

    transform = A.Compose(
        [
            A.Resize(height=256, width=128),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_dataset = ProstateCancerDataset(TEST_IMG_DIR, TEST_SEG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Visualize predictions
    visualize_predictions(test_loader, model, device=DEVICE, num_images=3)
    check_accuracy(test_loader, model, DEVICE)



if __name__ == "__main__":
    predict()
    