import torch
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import matplotlib.pyplot as plt
import nibabel as nib

class HipMRIDataset(Dataset):
    def __init__(self, dataset_root: str, train: bool = True, concise: bool = False, device: torch.device = 'cpu', transform = None):
        super().__init__()
        TRAIN_DIR, TRAIN_SEGMENT_DIR = 'keras_slices_train', 'keras_slices_seg_train'
        TEST_DIR, TEST_SEGMENT_DIR = 'keras_slices_test', 'keras_slices_seg_test'

        image_dir = os.path.join(dataset_root, TRAIN_DIR)
        segment_dir = os.path.join(dataset_root, TRAIN_SEGMENT_DIR)
        if not train:
            image_dir = os.path.join(dataset_root, TEST_DIR)
            segment_dir = os.path.join(dataset_root, TEST_SEGMENT_DIR)

        image_names = sorted(os.listdir(image_dir))
        segment_names = sorted(os.listdir(segment_dir))

        if len(image_names) != len(segment_names) or not all(a[4:] == b[3:] for a, b in zip(image_names, segment_names)):
            raise RuntimeError('The files have different names!')

        if concise:
            image_names = image_names[:min(20, len(image_names))]
            segment_names = segment_names[:min(20, len(segment_names))]

        self.image_items = list(map(lambda f : os.path.join(image_dir, f), image_names))
        self.segment_items = list(map(lambda f : os.path.join(segment_dir, f), segment_names))
        self.device = device
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = nib.load(self.image_items[index])
        image_data = image.get_fdata(caching='unchanged')
        image_tensor = torch.from_numpy(image_data)[:, :128].unsqueeze(0).to(self.device).to(torch.float32)

        segment = nib.load(self.segment_items[index])
        segment_data = segment.get_fdata(caching='unchanged')
        segment_tensor = torch.from_numpy(segment_data)[:, :128].unsqueeze(0).to(self.device).to(torch.float32)
        PROSTATE_INDEX = 5
        segment_tensor = (segment_tensor == PROSTATE_INDEX).float()

        if self.transform is not None:
            image_tensor = self.transform(image_tensor.float())

        return image_tensor, segment_tensor


DATASET_ROOT = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = HipMRIDataset(DATASET_ROOT, concise=True, device=device)
    images, segments = next(iter(DataLoader(dataset, batch_size=20, shuffle=True)))

    fig, axes = plt.subplots(4, 5)

    fig.set_size_inches(12, 8)
    fig.set_dpi(1000)
    fig.tight_layout()

    for i, axis in enumerate(axes.ravel()):
        axis.set_title(f'Example {i + 1}')
        axis.imshow(torch.swapaxes(images[i], -1, -2).squeeze().flip(0).cpu(), cmap='gist_gray')
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig('exampleimages.png')
    fig.clf()

    fig, axes = plt.subplots(4, 5)

    fig.set_size_inches(12, 8)
    fig.set_dpi(1000)
    fig.tight_layout()

    for i, axis in enumerate(axes.ravel()):
        axis.set_title(f'Example {i + 1}')
        axis.imshow(torch.swapaxes(segments[i], -1, -2).squeeze().flip(0).cpu(), cmap='gist_gray')
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig('examplesegments.png')
    fig.clf()
