import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules import UNet
from dataset import MedicalImageDataset
from train import dice_coefficient

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available, using CPU.")

model = tf.keras.models.load_model('unet_model', custom_objects={'dice_coefficient': dice_coefficient})

test_image_dir = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/keras_slices_test"
test_dataset = MedicalImageDataset(image_dir=test_image_dir, normImage=True, batch_size=4, shuffle=False)
test_loader = test_dataset.get_dataset()


def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)
    return dice


def test_model(model, test_loader):
    print("> Testing")
    dice_scores = []
    
    for images, labels in test_loader:
        predicted_masks = model.predict(images)
        predicted_masks = (predicted_masks > 0.5).astype(np.float32)
        
        for pred, target in zip(predicted_masks, labels):
            dice = dice_coefficient(target, pred)
            dice_scores.append(dice)
    
    avg_dice = np.mean(dice_scores)
    print(f'Average Dice Coefficient: {avg_dice:.4f}')
    return dice_scores

def visualize_predictions(model, test_loader, num_images=4):
    print("> Visualizing Predictions")
    
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    predicted_masks = model.predict(images)
    predicted_masks = (predicted_masks > 0.5).astype(np.float32)
    
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))
    
    for i in range(min(num_images, len(images))):
        ax_image = axes[i, 0]
        ax_image.imshow(images[i].squeeze(), cmap='gray')
        ax_image.set_title(f'Image {i+1}')
        ax_image.axis('off')
        
        ax_gt = axes[i, 1]
        ax_gt.imshow(labels[i].squeeze(), cmap='gray')
        ax_gt.set_title(f'Ground Truth {i+1}')
        ax_gt.axis('off')
        
        ax_pred = axes[i, 2]
        ax_pred.imshow(predicted_masks[i].squeeze(), cmap='gray')
        ax_pred.set_title(f'Prediction {i+1}')
        ax_pred.axis('off')
    
    plt.tight_layout()
    plt.show()

dice_scores = test_model(model, test_loader)
visualize_predictions(model, test_loader, num_images=4)
