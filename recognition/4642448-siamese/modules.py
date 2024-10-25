"""

@file modules.py
@author Matt Hoffman
@date 24/10/2024 (lmao)

pylint modules.py

-------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 8.86/10, +1.14)

"""

import torch
from torch import zeros, mean, clamp
from torch.nn import Module, Conv2d, ModuleList, Flatten, Linear, ReLU, MaxPool2d
from torch.nn.functional import pairwise_distance

class CompareAndContrast(Module):

	""" Module to compare based on euclidean distance
		which is fairly standard for siamese networks """

	def __init__(self, marge=2.0):

		""" init, innit? """

		super().__init__()

		self._marge = marge
		self._dim = True
		self._min_clamp = 0.0

	def forward(self, x, y, label):

		""" calculate the mean euclidean distance between
			two inputs """

		euc_dist = pairwise_distance(x, y, keepdim=self._dim)
		clamp_input = self._marge - euc_dist

		input_to_mean = (1 - label)*torch.pow(euc_dist, 2) + \
			label*torch.pow(clamp(clamp_input, min=self._min_clamp), 2)

		return mean(input_to_mean)

class SiameseCNN(Module):

	""" The CNN that will make the base for our
		network, to feed various inputs to. """

	def __init__(self, shape, class_count=2):

		""" Set up the CNN backbone for the network """

		super().__init__()

		self._my_dims = (1, 3)
		self._gp_flatten = Flatten()

		self._my_internal_nn = ModuleList([

			Conv2d(3, 96, kernel_size=11, stride=4),
			# ReLU activation layers might try leaky at some point
			ReLU(inplace=True),
			MaxPool2d(3, stride=2),

			Conv2d(96, 256, kernel_size=5, stride=1),
			# ReLU activation layers might try leaky at some point
			ReLU(inplace=True),
			MaxPool2d(2, stride=2),

			Conv2d(256, 384, kernel_size=3, stride=1),
			# ReLU activation layers might try leaky at some point
			ReLU(inplace=True)
		])

		self._gp_finisher = ModuleList([

			Linear(self.sizeof_shape(shape), 1024),
			ReLU(inplace=True),

			Linear(1024, 256),
			ReLU(inplace=True),
			Linear(256, class_count)
		])

	def forward(self, ix):

		""" Run an input through the CNN """

		for layer in self._my_internal_nn:

			ix = layer(ix)

		ix = self._gp_flatten(ix)

		for layer in self._gp_finisher:
			ix = layer(ix)

		return ix

	def sizeof_shape(self, shape):

		""" intermediary step for linear layer """

		ix = zeros(self._my_dims[0], self._my_dims[1], *shape)

		for layer in self._my_internal_nn:
			ix = layer(ix)

		ix = self._gp_flatten(ix)

		return ix.numel()

class TheTwins(Module):

	""" Pretty simple concept just add an
		interface to plug the outputs together """

	def __init__(self):

		""" set up the inner CNN """

		super().__init__()

		self._my_target_shape = (256, 256)
		self._my_cnn = SiameseCNN(self._my_target_shape)

	def get_cnn(self):

		""" does what it says on the tin """

		return self._my_cnn

	def forward(self, x, y):

		""" run the dual inputs through the CNN """

		return self._my_cnn.forward(x), self._my_cnn.forward(y)
