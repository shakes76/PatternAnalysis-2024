"""
Contains data processing and visualization functions for StyleGAN2.
"""
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from config import RANGPUR_PATH, PATH, ADNI_IMG_SIZE, ADNI_TRAIN_PATH, ADNI_TEST_PATH
from config import CIFAR_PATH, CIFAR_IMG_SIZE

def process_adni(batch_size, rangpur=False):
    """
    Returns dataset and dataloader for ADNI (For Image Generation).
    """
    transform = transforms.Compose([
        transforms.Resize((ADNI_IMG_SIZE, ADNI_IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Determine the correct path based on the rangpur flag
    base_path = RANGPUR_PATH if rangpur else PATH
    train_set = datasets.ImageFolder(base_path + ADNI_TRAIN_PATH, transform=transform)
    test_set = datasets.ImageFolder(base_path + ADNI_TEST_PATH, transform=transform)

    dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataset, dataloader

def process_cifar(batch_size, rangpur=False):
    """
    Returns dataset and dataloader for CIFAR10 (For Image Generation).
    """
    transform = transforms.Compose([
        transforms.Resize((CIFAR_IMG_SIZE, CIFAR_IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Determine the correct path based on the rangpur flag
    base_path = RANGPUR_PATH if rangpur else PATH
    train_set = datasets.CIFAR10(base_path + CIFAR_PATH, transform=transform, train=True, download=True)
    test_set = datasets.CIFAR10(base_path + CIFAR_PATH, transform=transform, train=True, download=True)

    dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataset, dataloader

def visualize_dataset_embeddings(dataloader, num_samples=1000):
    """
    Creates TSNE and UMAP visualizations of the dataset from a dataloader

    Args:
        dataloader: PyTorch DataLoader containing the dataset
        num_samples: Number of samples to use for visualization (default: 1000)

    Acknowledgement:
        https://plotly.com/python/t-sne-and-umap-projections/
        LLM: Claude 3.5 Sonnet
    """
    # Collect samples and labels
    data = []
    labels = []
    # ADNI Classes: Alzheimer's Disease (AD) and Normal Control (NC)
    classes = ['AD', 'NC']
    sample_count = 0

    # Iterate through dataloader to collect samples
    for batch, label in dataloader:
        batch_size = batch.size(0)
        remaining = num_samples - sample_count

        # Take only what we need from this batch
        samples_to_take = min(batch_size, remaining)

        # Convert images to numpy and reshape
        batch_numpy = batch[:samples_to_take].cpu().numpy()
        # Reshape from (N, C, H, W) to (N, C*H*W)
        batch_flat = batch_numpy.reshape(samples_to_take, -1)

        data.append(batch_flat)
        labels.extend(label[:samples_to_take].cpu().numpy())

        sample_count += samples_to_take
        if sample_count >= num_samples:
            break

    # Combine all batches
    data = np.concatenate(data, axis=0)
    labels = np.array(labels)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Calculate T-SNE embedding
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(scaled_data)

    # Calculate UMAP embedding
    print("Computing UMAP embedding...")
    reducer = umap.UMAP(random_state=42)
    umap_embedding = reducer.fit_transform(scaled_data)

    # Ensure embeddings are dense arrays
    tsne_embedding = np.asarray(tsne_embedding)
    umap_embedding = np.asarray(umap_embedding)

     # Set up colors for ADNI classes
    colors = ['#FF6B6B', '#4ECDC4'] # Red, Turquoise

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot T-SNE with distinct colors for each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        scatter1 = ax1.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                                label=classes[i], alpha=0.6, c=colors[i])

    ax1.set_title('t-SNE Embedding of ADNI Brain MRI Scans', fontsize=14)
    ax1.set_xlabel('First Component', fontsize=12)
    ax1.set_ylabel('Second Component', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Plot UMAP with distinct colors for each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        scatter2 = ax2.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                                label=classes[i], alpha=0.6, c=colors[i])

    ax2.set_title('UMAP Embedding of ADNI Brain MRI Scans', fontsize=14)
    ax2.set_xlabel('First Component', fontsize=12)
    ax2.set_ylabel('Second Component', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Add a title for the entire figure
    plt.suptitle('Distribution of Alzheimer\'s Disease vs Normal Control Brain MRI Scans',
                fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()
