import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import ProstateSegmentationDataset
from modules import UNet

# Dice coefficient
def dice_coefficient(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing dataset path (needs to be changed)
    test_image_dir = 'C:/Users/一七/Desktop/HipMRI_study_keras_slices_data/keras_slices_test'
    test_label_dir = 'C:/Users/一七/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_test'

    # Load testing dataset
    test_dataset = ProstateSegmentationDataset(test_image_dir, test_label_dir, norm_image=True, categorical=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('unet_model.pth')) 
    model.eval()

    dice_scores = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.float().to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            pred = (outputs > 0.5).float() 
            dice = dice_coefficient(pred, labels)
            dice_scores.append(dice.item())

            # Print Dice Coefficient
            print(f"Sample {i+1}: Dice Coefficient = {dice.item()}")

            # Visualize input images, label and predicted results
            '''
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Input Image')

            plt.subplot(1, 3, 2)
            plt.imshow(labels[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('True Mask')

            plt.subplot(1, 3, 3)
            plt.imshow(pred[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Predicted Mask')

            plt.show()
            '''

    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f"Average Dice Coefficient: {avg_dice}")

if __name__ == '__main__':
    predict()
