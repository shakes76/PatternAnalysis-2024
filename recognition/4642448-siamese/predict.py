"""

@file predict.py
@author Matt Hoffman
@date 25/10/2024 (lmao)

$ pylint predict.py

************* Module predict
predict.py:57:0: C0301: Line too long (116/100) (line-too-long)

------------------------------------------------------------------
Your code has been rated at 9.67/10 (previous run: 9.00/10, +0.67)

"""

import torch
from torch import device, cuda, load, no_grad

from torch.utils.data import DataLoader

from modules import TheTwins
from dataset import create_preset_dataloader
from train import SAVE_FILE

TRAIN_DEVICE = device("cuda" if cuda.is_available() else "cpu")

TARGET_SAMPLES = 1000
BATCH_SIZE = 32

if __name__ == "__main__":

	my_dataloader = DataLoader(create_preset_dataloader(dataset_type=1), batch_size=BATCH_SIZE, shuffle=False)

	# load up the model
	siam_net = TheTwins().to(TRAIN_DEVICE)
	siam_net.load_state_dict(load(SAVE_FILE, map_location=TRAIN_DEVICE))

	siam_net.get_cnn().eval()

	predict_stats = {

		"total": 0,
		"correct": 0
	}

	i = 0

	with no_grad():
		for image, label in my_dataloader:

			# dont want to test any more
			if i == TARGET_SAMPLES:
				break

			image, label = image.to(TRAIN_DEVICE), label.to(TRAIN_DEVICE)

			# run the test image through the trained cnn
			out = siam_net.get_cnn()(image)
			_, predicted = torch.max(out, 1)

			predict_stats["correct"] += (predicted == label).sum().item()
			predict_stats["total"] += label.size(0)

			i += 1

			acc = (predict_stats['correct'] / predict_stats['total']) * 100
			print(f"{i} / {TARGET_SAMPLES}, stats: {predict_stats['correct']} / {predict_stats['total']} - {acc}%", end="\r")

	acc = (predict_stats['correct'] / predict_stats['total']) * 100
	print(f'ISIC test data accuracy: {predict_stats['correct']} / {predict_stats['total']} - {acc}%')
