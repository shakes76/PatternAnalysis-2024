import torch
from train import train_model
from dataset import load_data_2D
from modules import dice_coefficient
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train_model().to(device)
model.eval()
batch_size = 32
def visualize_predictions():
    with torch.no_grad():
        test_images_filenames = [('keras_slices_seg_test/' + f) for f in os.listdir('keras_slices_seg_test') if
                                 f.endswith('.nii.gz')]
        test_labels_filenames = [('keras_slices_test/' + f) for f in os.listdir('keras_slices_test') if
                                 f.endswith('.nii.gz')]
        test_images = load_data_2D(test_images_filenames, normImage=True)
        test_labels = load_data_2D(test_labels_filenames, normImage=True)
        for i in range(0, len(test_images), batch_size):
            images = test_images[i:i + batch_size]
            labels = test_labels[i:i + batch_size]
            images = np.expand_dims(images, axis=1).astype(np.float16)
            labels = np.expand_dims(labels, axis=1).astype(np.float16)
            images = torch.from_numpy(images).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            predicted = outputs > 0.5
        dice_score = f1_score(test_labels.cpu().flatten(), predicted.cpu().flatten())
        print(f"Dice Similarity Score on Test Set: {dice_score:.4f}")
        num_samples_to_show = 5
        fig, ax = plt.subplots(num_samples_to_show, 3, figsize=(15, num_samples_to_show * 5))
        for i in range(num_samples_to_show):
            # Original image
            ax[i, 0].imshow(test_images[i].cpu().squeeze(), cmap='gray')
            ax[i, 0].set_title("Original Image")
            ax[i, 0].axis("off")
            # Ground truth label
            ax[i, 1].imshow(test_labels[i].cpu().squeeze(), cmap='gray')
            ax[i, 1].set_title("Ground Truth Label")
            ax[i, 1].axis("off")
            # Model Prediction
            ax[i, 2].imshow(predicted[i].cpu().squeeze(), cmap='gray')
            ax[i, 2].set_title("Model Prediction")
            ax[i, 2].axis("off")
        plt.tight_layout()
        plt.show()


visualize_predictions()

def evaluate_dice():
    dice_scores = []
    with torch.no_grad():
        test_images_filenames = [('keras_slices_seg_test/' + f) for f in os.listdir('keras_slices_seg_test') if
                                 f.endswith('.nii.gz')]
        test_labels_filenames = [('keras_slices_test/' + f) for f in os.listdir('keras_slices_test') if
                                 f.endswith('.nii.gz')]
        test_images = load_data_2D(test_images_filenames, normImage=True)
        test_labels = load_data_2D(test_labels_filenames, normImage=True)
        for i in range(0, len(test_images), batch_size):
            images = test_images[i:i + batch_size]
            labels = test_labels[i:i + batch_size]
            images = np.expand_dims(images, axis=1).astype(np.float16)
            labels = np.expand_dims(labels, axis=1).astype(np.float16)
            images = torch.from_numpy(images).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, labels)
            dice_scores.append(dice.item())

    avg_dice = np.mean(dice_scores)
    print(f'Average Dice Similarity Coefficient on test set: {avg_dice:.4f}')