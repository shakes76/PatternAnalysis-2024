import os
import shutil
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt


def split_val_set(original_dataset_path, new_dataset_path, split_ratio):
    """
    split the orginal train set to new train set and val set by split ratio. Then save the new dataset.
    """
    print("splitting dataset")
    split_ratio = 1 - split_ratio
    # Define paths
    original_dataset_path = original_dataset_path      # the data path of the original data
    new_dataset_path = new_dataset_path    # the data path to store the new data

    train_path = os.path.join(original_dataset_path, 'train')
    test_path = os.path.join(original_dataset_path, 'test')

    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)

    # Create new directory structure
    os.makedirs(new_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'test'), exist_ok=True)

    # Define the classes
    classes = os.listdir(train_path)

    for class_name in classes:
        # Create class directories in the new dataset
        os.makedirs(os.path.join(new_dataset_path, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_path, 'val', class_name), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_path, 'test', class_name), exist_ok=True)

        # Get all images in the current class
        class_path = os.path.join(train_path, class_name)
        images = os.listdir(class_path)

        # Shuffle and split the images
        # random.shuffle(images)
        train_size = int(split_ratio * len(images))
        train_images = images[:train_size]
        val_images = images[train_size:]

        # Copy images to the new train and validation folders
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(new_dataset_path, 'train', class_name, img))

        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(new_dataset_path, 'val', class_name, img))

        # Copy test images (unchanged)
        class_test_path = os.path.join(test_path, class_name)
        for img in os.listdir(class_test_path):
            shutil.copy(os.path.join(class_test_path, img), os.path.join(new_dataset_path, 'test', class_name, img))

    print("Dataset split completed!")


def load_model(model, optimizer, filepath):
    """
    load the model parameters.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded model and optimizer from {filepath}")


def append_training_log(train_loss, train_acc, val_loss, val_acc, run_time, file_name="training_log.csv"):
    file_exists = os.path.isfile(file_name)

    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['Index', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'Run Time'])

        # Calculate current index (row number)
        index = sum(1 for _ in open(file_name)) - 1  # Subtracting 1 for the header row

        writer.writerow([index, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", run_time])


def draw_training_log(file_name="training_log.csv"):
    data = pd.read_csv(file_name)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(data.index, data['train_acc'], label='Train Acc', marker='o')
    plt.plot(data.index, data['val_acc'], label='Val Acc', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(data.index, data['train_loss'], label='Train Loss', marker='o', color='red')
    plt.plot(data.index, data['val_loss'], label='Val Loss', marker='o', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_process', dpi=600)

    plt.show()
