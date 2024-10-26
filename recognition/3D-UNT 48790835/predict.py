import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import Medical3DDataset, get_transform
import modules
import train
import os
import time

# Define paths for test data
TEST_IMAGES_PATH = "/Users/qiuhan/Desktop/UQ/3710/Improved-UNET-s4879083/重新下载的数据集/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
TEST_LABELS_PATH = "/Users/qiuhan/Desktop/UQ/3710/Improved-UNET-s4879083/重新下载的数据集/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon"

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")

    # Load test dataset
    testDataSet = Medical3DDataset(TEST_IMAGES_PATH, TEST_LABELS_PATH, get_transform())
    testDataloader = DataLoader(testDataSet, batch_size=train.batchSize, shuffle=False)

    # Load trained model
    model = modules.Improved2DUnet()
    model.load_state_dict(torch.load(train.modelPath))
    model.to(device)
    print("Model successfully loaded.")

    # Perform testing
    test(testDataloader, model, device)

def test(dataLoader, model, device):
    losses_validation = []
    dice_similarities_validation = []

    print("> Test inference started")
    start = time.time()
    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(dataLoader):
            images = images.to(device)
            labels = labels.to(device)

            # Get model outputs
            outputs = model(images)
            losses_validation.append(train.dice_loss(outputs, labels).item())
            dice_similarities_validation.append(train.dice_coefficient(outputs, labels).item())

            # Save segmentations for the first batch
            if step == 0:
                train.save_segments(images, labels, outputs, numComparisons=9, test=True)

        print(f'Test Loss: {train.get_average(losses_validation):.5f}, '
              f'Test Average Dice Similarity: {train.get_average(dice_similarities_validation):.5f}')
    end = time.time()
    elapsed = end - start
    print(f"Test inference took {elapsed/60:.2f} minutes in total")

if __name__ == "__main__":
    main()
