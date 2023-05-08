import os
import glob
from typing import Union, List, Iterable, Dict
from collections import deque
from enum import Enum 

import numpy as np
import torch
import torch.nn as nn

from Decorators.ClassDecorators import extend_super, extendable, ScopedRetValue
from dataclasses import dataclass

@dataclass 
class ModelStats: 
    episode: int 
    exploration_rate: float
    curr_step: int


class RLModelWrapperBase:

    CHECKPOINT_FILE_EXT = '.chkpt'
    def __init__(self, model: nn.Module, info: Union[Dict, None] = None):
        if info is None: 
            info = {}
        self.state_dim = info.get("state_dim")
        self.action_dim = info.get("action_dim")
        self.save_dir = info.get("save_dir", None)
        self.burnin = info.get("burnin", 1e4)          # num experiences before training
        self.learn_every = info.get("learn_every", 3)  # num experiences btwn updates to network
        self.sync_every = info.get("sync_every", 1e4)  # num experiences btwn network.sync calls
        self.save_every = info.get("save_every", 5e5)  # num experiences btwn saving model checkpoint

        self.learning_rate = info.get("learning_rate", 0.00025)
        # TODO: Implement Exploration vs Exploitation Strategies. Using \epsilon-greedy for the moment
        self.exploration_rate = info.get("exploration_rate", 1.0)
        self.exploration_rate_decay = info.get("exploration_rate_decay", 0.99999975)
        self.exploration_rate_min = info.get("exploration_rate_min", 0.1)
        self.curr_step = 0
        self.episodes = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = model.float().to(device = self.device)
        
        #self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.loss_fn = torch.nn.SmoothL1Loss()

        if self.save_dir is not None and not os.path.exists(self.save_dir):
            self.save_dir.mkdir(parents=True)

        self.save_attributes = ["state_dim", "action_dim", "save_dir", "burnin", 
                                "learn_every", "sync_every", "save_every", "learning_rate",
                                "exploration_rate", "exploration_rate_decay", "exploration_rate_min", 
                                "curr_step", "episodes"]

    def save_attr_dict(self):
        return {attr: getattr(self, attr) for attr in self.save_attributes}

    def save(self) -> None:
        if self.save_dir is None: 
            return  
        save_path = (
            self.save_dir / f"net_{int(self.curr_step // self.save_every)}{RLModelWrapperBase.CHECKPOINT_FILE_EXT}"
        )
        torch.save(
            dict(
                net_state_dict=self.net.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                save_attributes=self.save_attr_dict()
                ),
                save_path,
        )
        print(f"Network saved to {save_path} at step {self.curr_step}")

    def get_checkpoint(self, checkpoint_name: Union[str, None] = None):
        # TODO: Change to proper error 
        checkpoint_dir = self.save_dir
        if checkpoint_dir is None: 
            runs = glob.glob(f"checkpoints/*")
            if len(runs) == 0: 
                raise RuntimeError("No runs found in checkpoints/*")
            checkpoint_dir = max(runs, key = os.path.getctime)
        if checkpoint_name is None: 
            checkpoint_paths = glob.glob(f"{checkpoint_dir}/*{RLModelWrapperBase.CHECKPOINT_FILE_EXT}")
            if len(checkpoint_paths) == 0: 
                raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")
            path = max(checkpoint_paths, key = os.path.getctime)

        print("Loading checkpoint", path)
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for attrname, attrval in checkpoint["save_attributes"].items(): 
            setattr(self, attrname, attrval)


    def end_episode(self) -> ModelStats: 
        stats = ModelStats(episode = self.episodes, 
            exploration_rate = self.exploration_rate, 
            curr_step = self.curr_step)
        self.episodes += 1
        return stats 

    @extendable(returns_scope = False)
    def learn(self): 
        if self.curr_step % self.sync_every == 0:
            # Will raise AttibuteError if not syncable. 
            self.net.sync()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

    def act(self, state, exploit_only: bool = False):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action will perform
        """
        # Explore or Exploit 
        if np.random.rand() < self.exploration_rate and exploit_only is False:
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
    class SampleStrategy(Enum):
        Basic = "basic"
        # RankPriority = "rankpriority" # Can implement but it might be slow 
        ProportionalPriority = "priority"

    def __init__(self, model: nn.Module, info: Union[Dict, None] = None):
        super().__init__(model, info)
        self.memory_capacity = info.get("memory_capacity", 15000) # 100,000
        self.batch_size = info.get("batch_size", 32)
        self.memory = deque(maxlen=self.memory_capacity)
        self.sample_strategy = info.get("replay_sample_strategy", "basic")
        self.sample_strategy = RLModelWrapperReplay.SampleStrategy(self.sample_strategy)
        self.sample_dist = deque(maxlen=self.memory_capacity)
        self.sample_norm = 0
        self.sample_epsilon = info.get("replay_sample_epsilon", .001)

        # Saving and loading will empty memory. We're not going to save that 
        self.save_attributes += ["memory_capacity", "batch_size", "memory", 
            "sample_strategy", "sample_dist", "sample_norm", "sample_epsilon"]
 
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

        sample_weight = self.get_sample_weights(state.unsqueeze(0), next_state.unsqueeze(0), 
                                                action.unsqueeze(0), reward.unsqueeze(0), done.unsqueeze(0))[0].item()
        self.sample_norm += sample_weight  
        if len(self.memory) == self.memory_capacity: 
            self.sample_norm -= self.sample_dist[0]
        self.sample_dist.append(sample_weight)

        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        # TODO: self.sample_norm == 0
        batch_inds = np.random.choice(len(self.memory), self.batch_size, p = np.array(self.sample_dist) / self.sample_norm)
        batch = [self.memory[ind] for ind in batch_inds]
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return batch_inds, state, next_state, action, reward, done

    @extendable(returns_scope = False)
    def get_sample_weights(self, state, next_state, action, reward, done) -> float:
        # print("=========================")
        # print(f"{state.shape}")
        # print(f"{next_state.shape}")
        # print(f"{action.shape}")
        # print(f"{reward.shape}")
        # print(f"{done.shape}")
        if self.sample_strategy == RLModelWrapperReplay.SampleStrategy.Basic:
            return torch.ones(state.shape[0])
        
    def update_sample_dist(self, batch_inds: Iterable[int], weights: Iterable[float]) -> None: 
        for ind, weight in zip(batch_inds, weights):
            self.sample_norm -= self.sample_dist[ind]
            self.sample_dist[ind] = abs(weight.detach().cpu().item())
            self.sample_norm += self.sample_dist[ind]

    @extendable(returns_scope = True)
    @extend_super(RLModelWrapperBase)
    def learn(self, scope: Union[Dict, None] = None): 
        batch, state, next_state, action, reward, done = self.recall() # Sample batch from replay 
        scope = dict(
            batch_inds = batch, 
            state = state, 
            next_state = next_state, 
            action = action, 
            reward = reward, 
            done = done)
        return ScopedRetValue(value = None, scope = scope) 


class TDWrapper(RLModelWrapperReplay):
    def __init__(self, model: nn.Module, info: Union[Dict, None] = None):
        super().__init__(model, info)
        self.discount = info.get("discount", 0.9) # reward discount
        self.save_attributes += ["discount"]

    def td_estimate(self, state, action):
        # Online Q estimate = Q_online(s,a)
        current_Q = self.net(state, model = "online")
        current_v = current_Q.gather(1, action)
        return current_v

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        assert reward.shape[0] == next_state.shape[0] == done.shape[0], f"{reward.shape=}, {next_state.shape=} {done.shape=}"
        # TODO: Check Q and v follow literature 
        online_next_Q = self.net(next_state, model = "online")
        best_action = torch.argmax(online_next_Q, axis = 1).unsqueeze(1)
        bootstrap_next_Q = self.net(next_state, model = "target")
        bootstrap_next_v = bootstrap_next_Q.gather(1, best_action)
        discounted_next_v = self.discount * (1 - done.float()) * bootstrap_next_v
        return (reward + discounted_next_v).float()

    def update_parameters(self, td_estimate, td_target):
        # TODO: Move to RLModelWrapperBase
        assert td_target.shape == td_estimate.shape, f"{td_target.shape=} != {td_estimate.shape=}"
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        GRADIENT_CLIP = 100
        torch.nn.utils.clip_grad_value_(self.net.parameters(), GRADIENT_CLIP)
        self.optimizer.step()
        return loss.item()

    @extend_super(RLModelWrapperReplay, propogate_scope = False)
    def get_sample_weights(self, state, next_state, action, reward, done, scope: Union[Dict, None] = None):
        if self.sample_strategy == RLModelWrapperReplay.SampleStrategy.ProportionalPriority: 
            td_est = self.td_estimate(state, action)
            td_tgt = self.td_target(reward, next_state, done)
            sample_weights = abs(td_tgt - td_est) + self.sample_epsilon
            return sample_weights
        else: 
            raise ValueError(f"SampleStrategy ({self.sample_strategy}) is not supported")

    @extend_super(RLModelWrapperReplay, propogate_scope = False)
    def learn(self, scope: Union[Dict, None] = None):
        batch_inds = scope["batch_inds"]
        state = scope["state"]
        action = scope["action"]
        next_state = scope["next_state"]
        reward = scope["reward"]
        done = scope["done"]
        td_est = self.td_estimate(state = state, action = action)
        td_tgt = self.td_target(reward = reward, next_state = next_state, done = done)

        sample_weights = self.get_sample_weights(state, next_state, action, reward, done)
        self.update_sample_dist(batch_inds, sample_weights)
        
        loss = self.update_parameters(td_est, td_tgt)
        return (td_est.mean().item(), loss)




