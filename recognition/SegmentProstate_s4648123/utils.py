import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from typing import Sequence, Mapping
from torch.utils.data._utils.collate import default_collate

# user defined parameters
IMAGE_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
MASK_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
MODEL_PATH = "/home/Student/s4648123/MRI3/best_unet.pth"
RANDOM_SEED = 42


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Converts an array to one-hot encoded channels with a fixed number of classes.

    Parameters:
    - arr: Input array with categorical values.
    - num_classes: Total number of classes to ensure consistent channel encoding.
    - dtype: Data type for the output array.

    Returns:
    - One-hot encoded 4D NumPy array with a fixed number of channels.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c: c + 1][arr == c] = 1
    return res


def load_image_and_label_3D(image_file, label_file, dtype=np.float32):
    """
    Load a 3D medical image and its corresponding label file.
    Parameters:
    - image_file: Path to the medical image (non-categorical).
    - label_file: Path to the label file (categorical).
    - dtype: Data type for the output arrays.

    Returns:
    - image: 4D NumPy array (1, rows, cols, depth) for the input image.
    - label: 4D NumPy array (channels, rows, cols, depth) for the categorical label.
    """

    # Load the image data (non-categorical)
    nifti_image = nib.load(image_file)
    image = nifti_image.get_fdata(caching='unchanged').astype(dtype)
    if len(image.shape) == 4:
        image = image[:, :, :, 0]  # Remove extra dimensions if present
    # Add a channel dimension at the front: (1, rows, cols, depth)
    image = np.expand_dims(image, axis=0)

    # Load the label data (categorical)
    nifti_label = nib.load(label_file)
    label = nifti_label.get_fdata(caching='unchanged').astype(np.uint8)
    if len(label.shape) == 4:
        label = label[:, :, :, 0]  # Remove extra dimensions if present
    # Convert label to categorical (one-hot encoded) format
    label = to_channels(label, dtype=dtype)
    # Reorder label to (channels, rows, cols, depth)
    label = np.transpose(label, (3, 0, 1, 2))

    return image, label


def get_images():
    def extract_keys(file_path):
        parts = os.path.basename(file_path).split('_')
        return parts[0], str(parts[1])[-1]

    # List of image and mask filepaths
    image_files = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.endswith('.nii.gz')]
    mask_files = [os.path.join(MASK_DIR, fname) for fname in os.listdir(MASK_DIR) if fname.endswith('.nii.gz')]
    image_files, mask_files = sorted(image_files, key=extract_keys), sorted(mask_files, key=extract_keys)

    return np.array(image_files), np.array(mask_files)


def collate_batch(batch: Sequence):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    elem = batch[0]
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    collate_fn = default_collate
    if isinstance(elem, Mapping):
        batch_list = {}
        for k in elem:
            key = k
            data_for_batch = [d[key] for d in data]
            batch_list[key] = collate_fn(data_for_batch)
    else:
        batch_list = collate_fn(data)
    return batch_list


# # PLOTTING METHODS
def plot_and_save(x, y_data, labels, title, xlabel, ylabel, filename):
    plt.figure()
    for y, label in zip(y_data, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
import numpy as np
import matplotlib.pyplot as plt

def visualise_slices(images, targets, preds):
    # Get the batch size
    batch_size = images.shape[0]

    # Calculate the number of rows (3 slices for each image)
    num_slices = 3
    total_rows = num_slices * batch_size

    # Create a figure for all images
    fig, axes = plt.subplots(total_rows, 3, figsize=(15, 5 * total_rows))

    for i in range(batch_size):
        # Define the center slices for each dimension
        z_center = images.shape[4] // 2  # depth axis (fixed z-axis)
        y_center = images.shape[2] // 2  # height axis (fixed y-axis)
        x_center = images.shape[3] // 2  # width axis (fixed x-axis)

        # Extract and rotate slices for the input image
        image_slices = [
            np.rot90(images[i, 0, :, :, z_center]),  # x-y plane (fixed z-axis)
            np.rot90(images[i, 0, :, y_center, :]),  # x-z plane (fixed y-axis)
            np.rot90(images[i, 0, x_center, :, :]),  # y-z plane (fixed x-axis)
        ]

        # Convert one-hot encoded targets and predictions to categorical labels
        target_categorical = np.argmax(targets[i], axis=0)
        pred_categorical = np.argmax(preds[i], axis=0)

        # Extract and rotate slices for the target and prediction labels
        target_slices = [
            np.rot90(target_categorical[:, :, z_center]),  # x-y plane
            np.rot90(target_categorical[:, y_center, :]),  # x-z plane
            np.rot90(target_categorical[x_center, :, :]),  # y-z plane
        ]

        pred_slices = [
            np.rot90(pred_categorical[:, :, z_center]),  # x-y plane
            np.rot90(pred_categorical[:, y_center, :]),  # x-z plane
            np.rot90(pred_categorical[x_center, :, :]),  # y-z plane
        ]

        # Plot the slices
        for j in range(num_slices):
            # Original image with gray colormap
            axes[i * num_slices + j, 0].imshow(image_slices[j], cmap='gray')
            axes[i * num_slices + j, 0].set_title('Original Image')
            axes[i * num_slices + j, 0].axis('off')

            # Target label with turbo colormap
            axes[i * num_slices + j, 1].imshow(target_slices[j], cmap='turbo')
            axes[i * num_slices + j, 1].set_title('Label')
            axes[i * num_slices + j, 1].axis('off')

            # Prediction with turbo colormap
            axes[i * num_slices + j, 2].imshow(pred_slices[j], cmap='turbo')
            axes[i * num_slices + j, 2].set_title('Prediction')
            axes[i * num_slices + j, 2].axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_visualization.png', bbox_inches='tight')
    plt.close(fig)

    return

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

def animate_segmentation(images, predictions, filename='segmentation_animation.gif'):
    """
    Animate through an aerial view of the input image and the predicted segmentation.

    Args:
        images (torch.Tensor): Input images with shape (batch_size, channels, height, width, depth).
        predictions (torch.Tensor): Model predictions with shape (batch_size, classes, height, width, depth).
        filename (str): The name of the output GIF file.
    """
    # Get the number of slices and their dimensions
    num_slices = images.shape[4]  # Depth of the image
    num_classes = predictions.shape[1]  # Number of classes in the prediction

    # Create an array for grayscale predictions
    grayscale_predictions = np.zeros((predictions.shape[0], predictions.shape[2], predictions.shape[3], predictions.shape[4]))

    # Create grayscale images from one-hot encoding
    for i in range(predictions.shape[0]):
        grayscale_predictions[i] = np.argmax(predictions[i], axis=0)

    # Normalize grayscale predictions to range [0, 1] based on number of classes
    grayscale_predictions /= (num_classes - 1)

    # Set up the figure and axes
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))  # Increase width for better spacing
    for ax in axs:
        ax.axis('off')  # Turn off the axes

    def update(frame):
        for ax in axs:
            ax.clear()
            ax.axis('off')  # Turn off the axes again for each frame

        # Plot input image on the left (first channel) rotated 90 degrees
        axs[0].imshow(np.rot90(images[0, 0, :, :, frame]), cmap='gray', vmin=0, vmax=1)
        axs[0].set_title('Input Image', fontsize=12)

        # Plot the predicted segmentation on the right rotated 90 degrees
        axs[2].imshow(np.rot90(grayscale_predictions[0, :, :, frame]), cmap='gray', vmin=0, vmax=1)
        axs[2].set_title('Predicted Segmentation', fontsize=12)

        # Create an arrow that points from the input image to the predicted segmentation
        arrow = patches.FancyArrowPatch((0.3, 0.5), (0.7, 0.5), mutation_scale=20, color='black', lw=2)
        axs[1].add_patch(arrow)

        plt.tight_layout()  # Adjust layout to minimize padding

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_slices, repeat=False)

    # Save the animation as a GIF
    ani.save(filename, writer='pillow', fps=10)
    plt.close(fig)
    return



def animate_3d_segmentation(predictions, filename='3d_segmentation_animation.gif'):
    """
    Animate the last 4 classes of the predicted segmentation in a rotating invisible 3D plot.

    Args:
        predictions (torch.Tensor): Model predictions with shape (batch_size, classes, depth, height, width).
        filename (str): The name of the output GIF file.
    """
    last_4_classes = predictions[:, -4:, :, :, :]  # Select the last 4 classes

    # Get dimensions
    num_classes = last_4_classes.shape[1]  # Number of classes (4)
    depth, height, width = last_4_classes.shape[2], last_4_classes.shape[3], last_4_classes.shape[4]

    # Create a meshgrid for the 3D plot
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    z = np.linspace(0, depth - 1, depth)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Function to plot each class
    def plot_classes(ax, class_data):
        ax.clear()
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Define shades of gray for the classes
        gray_colors = [0.3, 0.5, 0.7, 0.9]  # Dark to light gray

        # Plot each class
        for i in range(num_classes):
            # Create a mask for the current class
            mask = class_data[i] > 0  # Boolean mask for the current class
            ax.scatter(x[mask], y[mask], z[mask], alpha=0.5, s=1, color=(gray_colors[i],) * 3, label=f'Class {i + 1}')

    # Update function for animation
    def update(frame):
        ax.view_init(elev=10, azim=frame)  # Rotate around the azimuth angle
        plot_classes(ax, last_4_classes[0])  # Plot classes for the first image in the batch
        return ax,

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), repeat=False)

    # Save the animation as a GIF
    ani.save(filename, writer='pillow', fps=10)
    plt.close(fig)
    return
