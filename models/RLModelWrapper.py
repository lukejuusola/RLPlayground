import os
from typing import Union, List, Iterable, Dict
from collections import deque

import numpy as np
import random
import torch
import torch.nn as nn

from Decorators.ClassDecorators import extend_super, extendable, ScopedRetValue

class RLModelWrapperBase:
    CHECKPOINT_FILE_EXT = '.chkpt'
    def __init__(self, model: nn.Module, info: Dict):
        self.state_dim = info.get("state_dim")
        self.action_dim = info.get("action_dim")
        self.save_dir = info.get("save_dir", None)
        self.burnin = info.get("burnin", 1e4)          # num experiences before training
        self.learn_every = info.get("learn_every", 3)  # num experiences btwn updates to network
        self.sync_every = info.get("sync_every", 1e4)  # num experiences btwn network.sync calls
        self.save_every = info.get("save_every", 5e5)  # num experiences btwn saving model checkpoint
        # TODO: Implement Exploration vs Exploitation Strategies. Using \epsilon-greedy for the moment
        self.exploration_rate = info.get("exploration_rate", 1.0)
        self.exploration_rate_decay = info.get("exploration_rate_decay", 0.99999975)
        self.exploration_rate_min = info.get("exploration_rate_min", 0.1)
        self.curr_step = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = model.float().to(device = self.device)
        
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True)

    def save(self) -> None:
        if self.save_dir is None: 
            return  
        save_path = (
            self.save_dir / f"net_{int(self.curr_step // self.save_every)}{RLModelWrapperBase.CHECKPOINT_FILE_EXT}"
        )
        # TODO: Add optimizer state to checkpoint to enable pausing training, also epoch.
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
                ),
                save_path,
        )
        print(f"Network saved to {save_path} at step {self.curr_step}")

    def get_checkpoint(self, path: Union[str, None] = None):
        if path is None: 
            runs = glob.glob(f"checkpoints/*")
            last_run = max(runs, key = os.path.getctime)
            checkpoint_paths = glob.glob(f"{last_run}/*{RLModelWrapperBase.CHECKPOINT_FILE_EXT}")
            print(checkpoint_paths)
            path = max(checkpoint_paths, key = os.path.getctime)
        print("Loading checkpoint", path)
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model']) # 'model_state_dict'
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         epoch = checkpoint['epoch']
    #         loss = checkpoint['loss']


    @extendable(returns_scope = False)
    def learn(self): 
        if self.curr_step % self.sync_every == 0:
            # Will raise AttibuteError if not syncable. 
            # TODO: 
            # Sync strategy -- there's also a soft sync. 
            # target_parameter = tau * target_parameter + (1 - tau) * online_parameter
            self.net.sync()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action will perform
        """
        # Explore or Exploit 
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # Decay exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx


class RLModelWrapperReplay(RLModelWrapperBase):
    def __init__(self, model: nn.Module, info: dict):
        super().__init__(model, info)
        self.memory_capacity = info.get("memory_capacity", 15000) 
        self.batch_size = info.get("batch_size", 32)
        self.memory = deque(maxlen=self.memory_capacity)

        if self.burnin < self.batch_size: 
            raise ValueError("Burn-in must be larger than batch_size")
        if self.memory_capacity < self.batch_size: 
            raise ValueError("memory_capacity must be larger than batch_size")

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    @extendable(returns_scope = True)
    @extend_super(RLModelWrapperBase)
    def learn(self, scope: Union[Dict, None] = None): 
        state, next_state, action, reward, done = self.recall() # Sample batch from replay 
        scope = dict(
            state = state, 
            next_state = next_state, 
            action = action, 
            reward = reward, 
            done = done)
        return ScopedRetValue(value = None, scope = scope) 


class TDWrapper(RLModelWrapperReplay):
    def __init__(self, model: nn.Module, info: Dict):
        super().__init__(model, info)
        self.discount = info.get("discount", 0.9) # reward discount
        self.learning_rate = info.get("lr", 0.00025)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def td_estimate(self, state, action):
        # Online Q estimate = Q_online(s,a)
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ] 
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # Bootstrapped Q estimate
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.discount * next_Q).float()

    def update_parameters(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @extend_super(RLModelWrapperReplay, propogate_scope = False)
    def learn(self, scope: Union[Dict, None] = None):
        state = scope["state"]
        action = scope["action"]
        next_state = scope["next_state"]
        reward = scope["reward"]
        done = scope["done"]
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_parameters(td_est, td_tgt)
        return (td_est.mean().item(), loss)




