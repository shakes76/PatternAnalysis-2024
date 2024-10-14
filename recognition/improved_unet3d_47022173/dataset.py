import numpy as np
import tqdm
import nibabel as nib
from utils import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchio as tio
import torch

if IS_RANGPUR:
	cutoff = 211
	end = 'nii.gz'
	split = 6
	length = 9
else:
	cutoff = 24
	end = 'nii'
	split = 3
	length = 14


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
	def __init__(self, images_path, masks_path, transforms, mode):
		image_names = [f.name for f in Path(images_path).iterdir() if f.is_file() and f.name.endswith(end)]
		mask_names = [f.name for f in Path(masks_path).iterdir() if f.is_file() and f.name.endswith(end)]
		
		# Sort to ensure correct matching of image to mask
		image_names.sort()
		mask_names.sort()

		image_names = list(map(lambda x: images_path + x, image_names))
		mask_names = list(map(lambda x: masks_path + x, mask_names))

		match mode:
			case "train":
				self.image_names = image_names[:VALID_START]
				self.mask_names = mask_names[:VALID_START]
			case "valid":
				self.image_names = image_names[VALID_START:TEST_START]
				self.mask_names = mask_names[VALID_START:TEST_START]
			case "test":
				self.image_names = image_names[TEST_START:]
				self.mask_names = mask_names[TEST_START:]
			case "debug":
				self.image_names = image_names[:DEBUG]
				self.mask_names = mask_names[:DEBUG]
			case _:
				raise ValueError(f"Invalid mode: {mode}")

		self.images = torch.empty(0, 128, 128, 128)
		self.masks = torch.empty(0, 128, 128, 128)
		# self.affine = torch.empty(0, 4, 4)

		# TODO REMOVE THIS. Checks for matching image and mask names
		for image, mask in zip(self.image_names, self.mask_names):
			if image.split('/')[split][:length] != mask.split('/')[split][:length]:
				print(image, mask)
				raise ValueError("Image and mask do not match")

		# Load and transform LOAD_SIZE (50) images at a time due to memory constraints
		loaded = 0
		while len(self.images) < len(self.image_names):
			raw_images = load_data_3D(self.image_names[loaded:loaded + LOAD_SIZE])
			raw_masks = load_data_3D(self.mask_names[loaded:loaded + LOAD_SIZE])
			# raw_masks, affine = load_data_3D(self.mask_names[loaded:loaded + LOAD_SIZE], getAffines=True)
			subject = tio.Subject(
				image=tio.ScalarImage(tensor=raw_images),
				mask=tio.LabelMap(tensor=raw_masks),
			)
			
			transformed = transforms(subject)
			self.images = torch.cat((self.images, transformed['image'].data), dim=0) # Batch, 128 ,128 ,128
			self.masks = torch.cat((self.masks, transformed['mask'].data), dim=0)
			# self.affine = torch.cat((self.affine, torch.tensor(affine)), dim=0)

			loaded = len(self.images)
			print(f"Loaded {len(self.images)} / {len(self.image_names)}")
    
	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx].unsqueeze(0) # Add channel dimension
		mask = self.masks[idx].unsqueeze(0)
		# affine = self.affine[idx]
		return image, mask#, affine