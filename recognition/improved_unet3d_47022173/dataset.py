import numpy as np
import tqdm
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchio as tio

def load_data_3D(imageNames, normImage=False, dtype=np.float32, 
				getAffines=False, early_stop=False):
	'''
	Load medical image data from names, cases list provided into a list for each.
	This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.
	normImage: bool (normalise the image 0.0-1.0)
	orient: Apply orientation and resample image? Good for images with large slice 
	thickness or anisotropic resolution
	dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is 
	masks
	early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick 
	loading and testing scripts.
	'''

	affines = []
	interp = 'linear'
	if dtype == np.uint8:  # assume masks
		interp = 'nearest'

	num = len(imageNames)
	niftiImage = nib.load(imageNames[0])

	first_case = niftiImage.get_fdata(caching='unchanged')

	if len(first_case.shape) == 4:
		first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
		rows, cols, depth, channels = first_case.shape
		images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
	else:
		rows, cols, depth = first_case.shape
		images = np.zeros((num, rows, cols, depth), dtype=dtype)

	for i, inName in enumerate(tqdm.tqdm(imageNames)):
		niftiImage = nib.load(inName)

		inImage = niftiImage.get_fdata(caching='unchanged')  # read from disk only
		affine = niftiImage.affine

		if len(inImage.shape) == 4:
			inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
		inImage = inImage[:, :, :depth]  # clip slices
		inImage = inImage.astype(dtype)

		if normImage:
			inImage = (inImage - inImage.mean()) / inImage.std()

		images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage  # with pad

		affines.append(affine)

		if i > 20 and early_stop:
			break

	if getAffines:
		return images, affines
	else:
		return images


class ProstateDataset3D(Dataset):
	def __init__(self, images_path, masks_path, transforms):
		self.image_names = [images_path + f.name for f in Path(images_path).iterdir() if f.is_file() and f.name.endswith('.nii')]
		self.mask_names = [masks_path + f.name for f in Path(masks_path).iterdir() if f.is_file() and f.name.endswith('.nii')]

		raw_images = load_data_3D(self.image_names[:10]) # TODO remove, load 10 for testing
		raw_masks = load_data_3D(self.mask_names[:10]) # TODO remove, load 10 for testing

		subject = tio.Subject(
			image=tio.ScalarImage(tensor=raw_images),
			mask=tio.LabelMap(tensor=raw_masks),
		)

		transformed = transforms(subject)
		self.images = transformed['image'].data
		self.masks = transformed['mask'].data
		
    
	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx].unsqueeze(0)
		mask = self.masks[idx].unsqueeze(0)
		return image, mask