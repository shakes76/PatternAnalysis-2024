import os
import shutil
import random


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
