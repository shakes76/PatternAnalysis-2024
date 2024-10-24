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
    checkpoint = torch.load(model_path)  # Load the checkpoint
    model.load_state_dict(checkpoint['state_dict'])  # Load model weights
    return model


# Function to run predictions
def predict():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = UNET(in_channels=1, out_channels=5).to(DEVICE)
    model = load_model("my_checkpoint.pth.tar", model)
    model.eval()  # Set the model to evaluation mode

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
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Visualize predictions (or you could run check_accuracy)
    visualize_predictions(test_loader, model, device=DEVICE, num_images=3)
    check_accuracy(test_loader, model, "cuda")
'''
# Visualize the results
def visualize_predictions(images, ground_truths, preds, num_images=3):
    for i in range(num_images):
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.title('Input Image')
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title('Ground Truth Mask')
        plt.imshow(ground_truths[i].cpu().argmax(dim=0), cmap='gray')  # Assuming one-hot encoding
        
        plt.subplot(1, 3, 3)
        plt.title('Predicted Mask')
        plt.imshow(preds[i].cpu(), cmap='gray')
        plt.show()
'''
#TEST_IMG_DIR = "C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_test"
#TEST_SEG_DIR = "C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_test"


if __name__ == "__main__":
    predict()
    '''
    # Load your test data
    test_dataset = ProstateCancerDataset(TEST_IMG_DIR, TEST_SEG_DIR)  # Change to TEST_IMG_DIR, TEST_SEG_DIR if you have them
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = load_model("my_checkpoint.pth.tar", device="cuda")
    
    check_accuracy(test_loader, model)
    visualize_predictions(test_loader, model, "cuda")


    # Run predictions
    #images, ground_truths, preds = predict(model, test_loader, device="cuda")
    
    # Visualize the results
    #visualize_predictions(images, ground_truths, preds)'''