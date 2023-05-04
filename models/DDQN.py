import torch
from torch import nn

import copy
from enum import Enum

class ModelType(Enum):
	Online = "online"
	Target = "target"

class DDQN(nn.Module):
	"""mini CNN structure
	input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output

	A la Rainbow
	"""
	def __init__(self, net: nn.Module):
		super().__init__()
		self.online = net
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