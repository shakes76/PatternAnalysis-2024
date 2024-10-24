"""

@file train.py
@author Matt Hoffman
@date 25/10/2024 (lmao)

welp - it trains
"""

from torch import device, save, cuda
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import create_preset_dataloader
from modules import TheTwins, CompareAndContrast

TRAIN_DEVICE = device("cuda" if cuda.is_available() else "cpu")
TARGET_LEARNING_RATE = 0.001

SAVE_FILE = "snet.pth"

if __name__ == "__main__":

	iterz = 0

	my_dataloader = DataLoader(create_preset_dataloader(),
		batch_size=32, shuffle=True)

	my_dl_size = len(my_dataloader)

	print(f"planning to run for - {my_dl_size} iterz")

	my_differ = CompareAndContrast()

	my_net = TheTwins().to(TRAIN_DEVICE)

	# going to use an Adam optimiser nothing crazy
	my_optimiser = optim.Adam(my_net.parameters(), lr=TARGET_LEARNING_RATE)

	for data in my_dataloader:

		# chuck them on my GPU

		x = data[0].to(TRAIN_DEVICE)
		y = data[1].to(TRAIN_DEVICE)

		lbls = data[2].to(TRAIN_DEVICE)

		my_optimiser.zero_grad()

		# run the images through the network
		# grab the difference between the two based on euclidean
		loss = my_differ(*my_net(x, y), lbls.float())
		loss.backward()

		my_optimiser.step()

		print(f"iter - {iterz} / {my_dl_size} == loss - {loss.item()}")

		iterz += 1

	print("done!")

	state = my_net.state_dict()
	save(state, SAVE_FILE)
