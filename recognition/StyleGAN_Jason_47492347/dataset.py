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
# def check_loader():
#     """
#     https://blog.paperspace.com/implementation-of-progan-from-scratch/
#     """
#     loader,_ = get_dataloader(128)
#     cloth ,_ = next(iter(loader))
#     _, ax    = plt.subplots(3,3, figsize=(8,8))
#     plt.suptitle('Some real samples', fontsize=15, fontweight='bold')
#     ind = 0 
#     for k in range(3):
#         for kk in range(3):
#             ind += 1
#             ax[k][kk].imshow((cloth[ind].permute(1,2,0)+1)/2) 
# check_loader()