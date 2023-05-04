import torch
from torch import nn

import copy
from enum import Enum
from typing import Union

class ModelType(Enum):
	Online = "online"
	Target = "target"

class SyncStrategy(Enum): 
	Hard = "hard"
	Soft = "soft"

class DDQN_Base(nn.Module):
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
		raise NotImplementedError("Virtual Function")


# TODO: This could also be done as a mode in the base class. Might refactor later. Depends how deep the inheritance goes.  
class DDQN_HardSync(DDQN_Base):
	def sync(self, tau: Union[float, None]):  
		self.target.load_state_dict(self.online.state_dict())

class DDQN_SoftSync(DDQN_Base):
	def __init__(self, net: nn.Module, sync_rate: float):
		super().__init__(net)
		if sync_rate < 0 or sync_rate > 1:
			raise ValueError("sink_rate must have 0 <= sink_rate <= 1")
		self.sync_rate = sync_rate

	def sync(self):  
		updated_state_dict = self.target.state_dict()
		online_state_dict = self.online.state_dict()
		for key in online_state_dict:
			online_part = self.sync_rate * online_state_dict[key]
			target_part = (1. - self.sync_rate) * updated_state_dict[key]
			updated_state_dict[key] = online_part + target_part 
		self.target.load_state_dict(updated_state_dict)



