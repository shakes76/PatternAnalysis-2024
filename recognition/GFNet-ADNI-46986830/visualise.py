import matplotlib.pyplot as plt
import torchvision
import math

def visualize_batch(data_loader, classes, num_images=500):
    """
    Visualize a batch of images from the DataLoader.
    
    Args:
        data_loader (DataLoader): The DataLoader to sample from.
        classes (list): List of class names (e.g., ["NC", "AD"]).
        num_images (int): Number of images to visualize.
    """
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    print(labels)

    # Define how many images to show
    num_images = min(num_images, len(images))

    # Calculate grid size (rows and columns)
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    # Make a grid of images
    img_grid = torchvision.utils.make_grid(images[:num_images], nrow=num_cols, padding=2)
    
    # Convert the image tensor back to a format Matplotlib can display
    img_grid = img_grid.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC

    # Denormalize the images (if normalization was used)
    # img_grid = img_grid * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    # img_grid = img_grid.clip(0, 1)

    # Plot the grid
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    plt.imshow(img_grid)
    plt.title('Batch from DataLoader')

    # Add labels for the images
    for i in range(num_images):
        label = classes[labels[i].item()]
        row = i // num_cols
        col = i % num_cols
        plt.text(col * 256, row * 240 + 20, label, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.8))
    
    plt.axis('off')
    plt.show()
