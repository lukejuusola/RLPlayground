import torch
from torch import nn

import copy
from enum import Enum

class ModelType(Enum):
	Online = "online"
	Target = "target"

class DoubleQCNN(nn.Module):
	"""mini CNN structure
	input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
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

		self.online = nn.Sequential(
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

		self.target = copy.deepcopy(self.online)

		# Q_target parameters are frozen.
		for p in self.target.parameters():
			p.requires_grad = False

	def forward(self, input, model):
		model_type = ModelType(model)
		if model_type == ModelType.Online:
			return self.online(input)
		elif model_type == ModelType.Target:
			return self.target(input)
		else: 
			raise ValueError(f"Model Type <{model}> could not be converted to DoubleQ.ModelType")

	def sync(self): 
		self.target.load_state_dict(self.online.state_dict())            