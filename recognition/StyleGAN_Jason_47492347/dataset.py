"""
Data processing and loading functionalities.
"""


from utils import *


def get_dataloader(img_size):
    """
    Returns a PyTorch DataLoader object containing batches of preprocessed
    images from the dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(CHANNELS_IMG),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ])
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    batch_size = BATCH_SIZES[int(log2(IMG_SIZE / 4))]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

loader, _ = get_dataloader(IMG_SIZE)

def check_loader():
    """
    https://blog.paperspace.com/implementation-of-progan-from-scratch/
    """
    loader, _ = get_dataloader(128)
    mri, _ = next(iter(loader))
    _, ax = plt.subplots(2, 4, figsize=(20, 5))
    plt.suptitle('Samples from dataloader', fontsize=15, fontweight='bold')
    for i in range(4):
        for j in range(2):
            ax[j, i].imshow(mri[j + i].permute(1, 2, 0).squeeze(), cmap='gray')
    plt.show()

check_loader()