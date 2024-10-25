"""

@file train.py
@author Matt Hoffman
@date 25/10/2024 (lmao)

welp - it trains

$ pylint train.py

-------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 9.72/10, +0.28)

"""

from datetime import datetime

from torch import device, save, cuda
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import create_preset_dataloader
from modules import TheTwins, CompareAndContrast

TRAIN_DEVICE = device("cuda" if cuda.is_available() else "cpu")

# Adam optimiser apparently likes 0.001 so i wont question
TARGET_LEARNING_RATE = 0.001
TARGET_EPOCHS = 2

SAVE_FILE = f"snet{TARGET_EPOCHS}.pth"

def do_epoch(dataloader, network, optimiser, diff):

	""" Trains a single epoch, can be re-called for multiple """

	iterz = 0

	dl_size = len(dataloader)

	for data in dataloader:

		# chuck them on my GPU

		x = data[0].to(TRAIN_DEVICE)
		y = data[1].to(TRAIN_DEVICE)

		lbls = data[2].to(TRAIN_DEVICE)

		optimiser.zero_grad()

		# run the images through the network
		# grab the difference between the two based on euclidean
		loss = diff(*network(x, y), lbls.float())
		loss.backward()

		optimiser.step()

		print(f"{datetime.now()} iter - {iterz} / {dl_size} == loss - {loss.item()}")

		iterz += 1

if __name__ == "__main__":

	# create preset just uses the default settings from dataset.py
	my_dataloader = DataLoader(create_preset_dataloader(),
		batch_size=32, shuffle=True)

	my_dl_size = len(my_dataloader)

	print(f"{datetime.now()} planning to run for - {my_dl_size} iterz on {TARGET_EPOCHS} epochs.")

	my_differ = CompareAndContrast()

	my_net = TheTwins().to(TRAIN_DEVICE)

	# going to use an Adam optimiser nothing crazy
	my_optimiser = Adam(my_net.parameters(), lr=TARGET_LEARNING_RATE)

	for i in range(0, TARGET_EPOCHS):

		print(f"{datetime.now()} EPOCH {i} / {TARGET_EPOCHS}")
		do_epoch(my_dataloader, my_net, my_optimiser, my_differ)

	print("done!")

	state = my_net.state_dict()
	save(state, SAVE_FILE)
