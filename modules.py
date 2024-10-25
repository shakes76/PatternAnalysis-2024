"""

@file modules.py
@author Matt Hoffman
@date 24/10/2024 (lmao)

"""

import torch
from torch import zeros, mean, clamp
from torch.nn import Module, Conv2d, ModuleList, Flatten, Linear, ReLU, MaxPool2d
from torch.nn.functional import pairwise_distance

class CompareAndContrast(Module):

	""" Module to compare based on euclidean distance
		which is fairly standard for siamese networks """

	def __init__(self, marge=2.0):

		super().__init__()

		self._marge = marge
		self._dim = True
		self._min_clamp = 0.0

	def forward(self, x, y, label):

		euc_dist = pairwise_distance(x, y, keepdim=self._dim)
		clamp_input = self._marge - euc_dist

		input_to_mean = (1 - label)*torch.pow(euc_dist, 2) + label*torch.pow(clamp(clamp_input, min=self._min_clamp), 2)

		return mean(input_to_mean)

class SiameseCNN(Module):

	""" The CNN that will make the base for our
		network, to feed various inputs to. """

	def __init__(self, shape, class_count=2):

		super().__init__()

		self._my_dims = (1, 3)

		self._gp_2d_pool = MaxPool2d(kernel_size=2, padding=1)

		self._gp_relu = ReLU()
		self._gp_flatten = Flatten()

		self._my_internal_nn = ModuleList([

			Conv2d(3, 64, kernel_size=10),
			Conv2d(64, 128, kernel_size=7),
			Conv2d(128, 128, kernel_size=4),
			Conv2d(128, 128, kernel_size=4),
		])

		self._gp_linear = Linear(self.sizeof_shape(shape), class_count)

	def forward(self, ix):

		for layer in self._my_internal_nn:

			ix = self._gp_relu(layer(ix))
			ix = self._gp_2d_pool(ix)

		ix = self._gp_flatten(ix)

		return self._gp_linear(ix)

	def sizeof_shape(self, shape):

		ix = zeros(self._my_dims[0], self._my_dims[1], *shape)

		for layer in self._my_internal_nn:

			ix = self._gp_relu(layer(ix))
			ix = self._gp_2d_pool(ix)

		return ix.numel()

class TheTwins(Module):

	""" Pretty simple concept just add an
		interface to plug the outputs together """

	def __init__(self):

		super().__init__()

		self._my_target_shape = (256, 256)
		self._my_cnn = SiameseCNN(self._my_target_shape)

	def forward(self, x, y):

		return self._my_cnn.forward(x), self._my_cnn.forward(y)
