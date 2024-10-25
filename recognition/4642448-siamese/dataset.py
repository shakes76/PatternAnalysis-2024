"""

@file dataset.py
@author Matt Hoffman
@date 24/10/2024 (lmao)

$ pylint dataset.py

-------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 9.84/10, +0.16)

"""

import os
import random

import pandas as pd
import kagglehub

import torch

from torch.utils.data import Dataset

from torchvision.transforms import Compose, Resize, ToTensor, \
	RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter

from PIL import Image

AUG_ROTATION = 30
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = AUG_BRIGHTNESS
AUG_SAT = AUG_BRIGHTNESS
AUG_HUE = AUG_BRIGHTNESS

class DataPairenator(Dataset):

	""" The goal is to find two images with
		the same label for our network to compare """

	def __init__(self, image_dir, label_file, image_transform=None, dataset_type=0):

		self.image_transform = image_transform
		self.image_dir = image_dir


		self.label_mapping = {row['isic_id'] + '.jpg': row['target'] \
			for key, row in pd.read_csv(label_file).iterrows()}

		self.image_names = []
		self.image_paths = []

		for dir_path, _, filenames in os.walk(self.image_dir):
			for filename in filenames:

				# if we dont have a label for it probably dont use it eh
				try:

					self.label_mapping[filename]
				except KeyError:

					continue

				if not (filename.endswith('.jpg') or filename.endswith('.jpeg')):
					continue

				self.image_names.append(filename)
				self.image_paths.append(os.path.join(dir_path, filename))

		self.dataset_type = dataset_type

		self.augment = Compose([

			RandomHorizontalFlip(),
			RandomVerticalFlip(),
			RandomRotation(AUG_ROTATION),

			ColorJitter(
				brightness=AUG_BRIGHTNESS,
				contrast=AUG_CONTRAST,
				saturation=AUG_SAT,
				hue=AUG_HUE
			),
		])

	def extract_image_label(self, image_filepath):
		""" Grab the image label associated with that file path """

		return self.label_mapping.get(os.path.basename(image_filepath), None)

	def get_an_augment(self, index):
		""" Generate an augmented (read: bad image) to classify """

		image = self.augment(Image.open(
			os.path.join(self.image_dir, self.image_names[index])).convert("RGB"))

		return self.image_transform(image), self.label_mapping[self.image_names[index]]

	def __getitem__(self, idx):

		""" Get an image pair if we're training or a
			potentially augmented if we're predicting """

		if self.dataset_type == 1:
			return self.get_an_augment(idx)

		image_a_path = random.choice(self.image_paths)
		image_a = self.image_transform(Image.open(image_a_path).convert("RGB"))

		label_a = self.extract_image_label(image_a_path)

		test_class = random.randint(0, 1)

		found_match = False

		while not found_match:

			image_b_path = random.choice(self.image_paths)
			label_b = self.extract_image_label(image_b_path)

			if test_class and label_a == label_b:
				found_match = True

			elif not test_class and label_a != label_b:
				found_match = True

		image_b = self.image_transform(Image.open(image_b_path).convert("RGB"))

		return image_a, image_b, torch.tensor(int(label_a != label_b), dtype=torch.float32)

	def __len__(self):

		""" does what it says on the tin """

		return len(self.image_paths)

def create_preset_dataloader(dataset_type=0):

	""" Utility function to get the base data with some presets """

	data_ref = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")

	image_dir = os.path.join(data_ref, "train-image/image")
	label_csv = os.path.join(data_ref, "train-metadata.csv")

	image_transform = Compose([

		Resize((256, 256)),
		ToTensor(),
	])

	return DataPairenator(image_dir=image_dir, label_file=label_csv, \
		image_transform=image_transform, dataset_type=dataset_type)

if __name__ == "__main__":

	paired_dataset = create_preset_dataloader()
