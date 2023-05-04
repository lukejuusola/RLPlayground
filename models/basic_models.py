from typing import Union, Dict 

import torch
import torch.nn as nn

class AtariCNN(nn.Module):
	"""mini CNN structure
	input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output

	A la Rainbow
	"""
	def __init__(self, input_dim, output_dim):
		super().__init__()
		c, h, w = input_dim

		# Used in gym's super mario bros env
		if h != 84:
		    raise ValueError(f"Expecting input height: 84, got: {h}")
		if w != 84:
		    raise ValueError(f"Expecting input width: 84, got: {w}")

		# TODO: Compute Conv2d output dimension based on input dimension. 

		self.net = nn.Sequential(
			nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, output_dim),
		)

	def forward(self, x):
		return self.net(x)

class FullyConnected(nn.Module):
	"""
	TODO: Cartpole should be easy and so we're going to solve it as a test to see if anything works
	"""
	def __init__(self, input_dim, output_dim, info: Union[Dict, None] = None):
		super().__init__()
		if info is None: 
			info = {}
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.n_hidden_layers = info.get("hidden_layers", 2)
		self.layer_size = info.get("layer_size", 256)

		if self.n_hidden_layers < 0: 
			raise ValueError("Network must have at least 0 layers (input and output)")

		self.layers = []
		self.layers.append(nn.Linear(input_dim, self.layer_size))
		self.layers.append(nn.ReLU())
		for _ in range(0, self.n_hidden_layers):
			self.layers.append(nn.Linear(self.layer_size, self.layer_size))
			self.layers.append(nn.ReLU())
		self.layers.append(nn.Linear(self.layer_size, output_dim))
		self.fc_layers = nn.Sequential(*self.layers)
	     
	def forward(self, x):
		return self.fc_layers(x)

