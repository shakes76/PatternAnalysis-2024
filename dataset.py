import os
import random

import pandas as pd
import kagglehub

import torch

from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image

class DataPairenator(Dataset):

	def __init__(self, image_dir, label_file, image_transform=None):

		self.image_transform = image_transform
		self.image_dir = image_dir

		self.image_paths = self.collect_image_paths()
		self.label_mapping = {row['isic_id'] + '.jpg': row['target'] for key, row in pd.read_csv(label_file).iterrows()}

	def collect_image_paths(self):

		image_files = []

		for dir_path, _, filenames in os.walk(self.image_dir):
			for filename in filenames:

				if not (filename.endswith('.jpg') or filename.endswith('.jpeg')):
					continue

				image_files.append(os.path.join(dir_path, filename))

		return image_files

	def extract_image_label(self, image_filepath):

		return self.label_mapping.get(os.path.basename(image_filepath), None)

	def __getitem__(self, idx):

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

		return len(self.image_paths)

if __name__ == "__main__":

	data_ref = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")

	image_dir = os.path.join(data_ref, "train-image/image")
	label_csv = os.path.join(data_ref, "train-metadata.csv")

	image_transform = transforms.Compose([

		transforms.Resize((256, 256)),
		transforms.ToTensor(),
	])

	paired_dataset = DataPairenator(image_dir=image_dir, label_file=label_csv, image_transform=image_transform)[1]
