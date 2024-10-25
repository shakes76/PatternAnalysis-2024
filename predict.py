"""

@file predict.py
@author Matt Hoffman
@date 25/10/2024 (lmao)

"""

import torch
from torch import device, cuda, load, no_grad

from torch.utils.data import DataLoader

from modules import TheTwins
from dataset import create_preset_dataloader
from train import SAVE_FILE

TRAIN_DEVICE = device("cuda" if cuda.is_available() else "cpu")

TARGET_SAMPLES = 100
BATCH_SIZE = 32

if __name__ == "__main__":

	my_dataloader = DataLoader(create_preset_dataloader(),
		batch_size=BATCH_SIZE, shuffle=True)

	siamese_model = TheTwins().to(TRAIN_DEVICE)
	siamese_model.load_state_dict(load(SAVE_FILE, map_location=TRAIN_DEVICE), strict=False)

	siamese_model._my_cnn.eval()

	predict_stats = {

		"total": 0,
		"correct": 0
	}

	i = 0

	with no_grad():
		for image_a, image_b, labels in my_dataloader:
			if i == TARGET_SAMPLES:

				break

			image_a, image_b, labels = image_a.to(TRAIN_DEVICE), image_b.to(TRAIN_DEVICE), labels.to(TRAIN_DEVICE)

			outputs = siamese_model._my_cnn(image_a)
			_, predicted = torch.max(outputs, 1)

			predict_stats["correct"] += (predicted == labels).sum().item()
			predict_stats["total"] += labels.size(0)

			i += 1

	accuracy = (predict_stats['correct'] / predict_stats['total']) * 100
	print(f'ISIC test data accuracy: {predict_stats['correct']} / {predict_stats['total']} - {accuracy}%')
