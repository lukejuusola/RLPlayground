from typing import Dict, Union
from .basic_models import AtariCNN, FullyConnected
from .DDQN import DDQN

import torch 
import torch.nn


def build_atari_cnn(state_dim, action_dim):
	return basic_models.AtariCNN(state_dim, action_dim)

def build_atari_dqnn(override_info: Union[Dict, None] = None):
	state_dim = info.get("state_dim", (4, 84, 84))
	cnn = build_atari_cnn(state_dim, info["action_dim"])
	return DDQN.DDQN(cnn)

def build_fc(in_dim: int, out_dim: int, info: Union[Dict, None] = None):
	return FullyConnected(in_dim, out_dim, info)

def build_fc_dqnn(in_dim: int, out_dim: int, info: Union[Dict, None] = None): 
	fc = build_fc(in_dim, out_dim, info)
	return DDQN(fc)

