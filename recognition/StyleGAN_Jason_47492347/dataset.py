from utils import *


def get_dataloader(img_size):
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

# TODO: Write unit test for dataloader
