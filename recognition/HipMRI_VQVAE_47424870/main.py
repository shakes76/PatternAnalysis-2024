from dataset import get_dataloader
import os
import matplotlib.pyplot as plt

def visualize_batch(batch, num_images=4):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, img in enumerate(batch[:num_images]):
        axes[i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.show()

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "keras_slices", "keras_slices_train")
    print(f"Data Directory: {data_dir}")

    dataloader = get_dataloader(root_dir=data_dir, batch_size=16, image_size=64, shuffle=True)

    for batch_idx, (images, _) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1} | Image Batch Shape: {images.shape}")
        if batch_idx == 0:  # Visualize only the first batch
            visualize_batch(images)
            break