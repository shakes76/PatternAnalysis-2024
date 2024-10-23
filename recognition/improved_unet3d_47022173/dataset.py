"""
This file contains the dataset class for the 3D U-Net model. The dataset class loads the images 
and masks from the specified path and applies the specified transforms.
The dataset class is used in the predict.py and train.py files to load the data and create the 
dataloaders.The train, validate, and test split is done manually per constants.

Abdullah Badat (47022173), abdullahbadat27@gmail.com
"""

import numpy as np
import tqdm
import nibabel as nib
from utils import *
from torch.utils.data import Dataset
from pathlib import Path
import torchio as tio
import torch


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
	"""
	Dataset class for 3D U-Net model. Loads images and masks from the specified 
	path and applies the specified transforms. Used to create dataloaders for the train,
	validate, and test splits.
	"""
	def __init__(self, images_path, masks_path, mode, transforms):
		"""
		Initialize the dataset class by loading the images and masks from the specified path.

		Parameters:
		images_path (str): Path to the images
		masks_path (str): Path to the masks
		mode (str): Mode to load the dataset (train, valid, test, debug)
		transform (torchio.transforms): Transforms to apply to the images and masks
		"""
		image_names = [f.name for f in Path(images_path).iterdir() if f.is_file() and
				  'nii' in f.name]
		mask_names = [f.name for f in Path(masks_path).iterdir() if f.is_file() and
				 'nii' in f.name]
		
		# Sort to ensure correct matching of image to mask
		image_names.sort()
		mask_names.sort()

		image_names = list(map(lambda x: images_path + x, image_names))
		mask_names = list(map(lambda x: masks_path + x, mask_names))

		# Train, validate, test split
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

		self.images = torch.empty(0, WIDTH, HEIGHT, DEPTH)
		self.masks = torch.empty(0, WIDTH, HEIGHT, DEPTH)
		self.affines = torch.empty(0, 4, 4)

		# Load and transform LOAD_SIZE (50) images at a time due to memory constraints
		loaded = 0
		while len(self.images) < len(self.image_names):
			raw_images = load_data_3D(self.image_names[loaded:loaded + LOAD_SIZE])
			raw_masks, affines = load_data_3D(self.mask_names[loaded:loaded + LOAD_SIZE],
									 getAffines=True)

			subject = tio.Subject(
				image=tio.ScalarImage(tensor=raw_images),
				mask=tio.LabelMap(tensor=raw_masks),
			)
			
			transformed = transforms(subject)
			# Batch, 128, 128, 128
			self.images = torch.cat((self.images, transformed['image'].data), dim=0) 
			self.masks = torch.cat((self.masks, transformed['mask'].data), dim=0)
			self.affines = torch.cat((self.affines, torch.tensor(np.array(affines))), dim=0)
	

			loaded = len(self.images)
			print(f"Loaded {len(self.images)} / {len(self.image_names)}")
		
		if self.masks.dtype != torch.long:
			self.masks = self.masks.long()
    
	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx].unsqueeze(0) # Add channel dimension
		mask = self.masks[idx]
		affines = self.affines[idx]
		return image, mask, affines