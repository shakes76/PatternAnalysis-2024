import torch
import os
import os.path
import matplotlib.pyplot as plt
import nibabel as nib

def load_images(file_names: list[str], concise: bool = False,
                         torch_device: torch.device = 'cpu') -> torch.Tensor:
    if concise:
        NUM_IMAGES_CONCISE = 20
        file_names = file_names[:min(NUM_IMAGES_CONCISE, len(file_names))]

    each_shape = nib.load(file_names[0]).get_fdata(caching='unchanged').shape
    data = torch.zeros((len(file_names),) + each_shape).to(torch_device)

    for i, file in enumerate(file_names):
        image = nib.load(file)
        image_data = image.get_fdata(caching='unchanged')
        data[i] = torch.from_numpy(image_data).to(torch_device)

    return data

def separate_prostate_channel(image: torch.Tensor) -> torch.Tensor:
    PROSTATE_INDEX = 5
    return (image == PROSTATE_INDEX).to(torch.uint8)

def load_hipmri(dataset_root: str, train: bool = True, concise: bool = False, torch_device: torch.device = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    TRAIN_DIR, TRAIN_SEGMENT_DIR = 'keras_slices_train', 'keras_slices_seg_train'
    TEST_DIR, TEST_SEGMENT_DIR = 'keras_slices_test', 'keras_slices_seg_test'

    image_dir = os.path.join(dataset_root, TRAIN_DIR)
    segment_dir = os.path.join(dataset_root, TRAIN_SEGMENT_DIR)
    if not train:
        image_dir = os.path.join(dataset_root, TEST_DIR)
        segment_dir = os.path.join(dataset_root, TEST_SEGMENT_DIR)

    image_filenames = list(map(lambda f : os.path.join(image_dir, f), os.listdir(image_dir)))
    segment_filenames = list(map(lambda f : os.path.join(segment_dir, f), os.listdir(segment_dir)))
    image_tensor = load_images(image_filenames, concise=concise, torch_device=torch_device)
    segment_tensor = load_images(segment_filenames, concise=concise, torch_device=torch_device)
    segment_tensor = separate_prostate_channel(segment_tensor)

    return image_tensor, segment_tensor

if __name__ == '__main__':
    # DATASET_LOCATION = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'
    DATASET_LOCATION = 'data/'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    images, segments = load_hipmri(DATASET_LOCATION, concise=True, torch_device=device)

    fig, axes = plt.subplots(4, 5)

    fig.set_size_inches(12, 8)
    fig.set_dpi(1000)
    fig.tight_layout()

    for i, axis in enumerate(axes.ravel()):
        axis.set_title(f'Example {i + 1}')
        axis.imshow(torch.swapaxes(images[i], -1, -2).flip(0).cpu(), cmap='gist_gray')
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
        axis.imshow(torch.swapaxes(segments[i], -1, -2).flip(0).cpu(), cmap='gist_gray')
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig('examplesegments.png')
    fig.clf()
