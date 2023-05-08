from typing import Dict, Union
from .basic_models import AtariCNN, FullyConnected
from .DDQN import SyncStrategy, DDQN_SoftSync, DDQN_HardSync

import torch 
import torch.nn

def build_atari_cnn(state_dim, action_dim):
	return AtariCNN(state_dim, action_dim)

def build_atari_dqnn(override_info: Union[Dict, None] = None):
	state_dim = override_info.get("state_dim", (4, 84, 84))
	cnn = build_atari_cnn(state_dim, override_info["action_dim"])
	return DDQN_HardSync(cnn)

def build_fc(in_dim: int, out_dim: int, info: Union[Dict, None] = None):
	return FullyConnected(in_dim, out_dim, info)

def build_fc_dqnn(in_dim: int, out_dim: int, info: Union[Dict, None] = None): 
	fc = build_fc(in_dim, out_dim, info)
	sync_type = SyncStrategy(info["sync_strategy"])
	if sync_type == SyncStrategy.Hard: 
		return DDQN_HardSync(fc)
	else: 
		return DDQN_SoftSync(fc, info["sync_rate"])

