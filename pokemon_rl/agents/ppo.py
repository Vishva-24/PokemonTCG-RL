import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Tuple, List, Optional

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

class ActorNetwork(nn.Module):
    """PPO Actor Network (Policy) for the Mini Pokémon TCG RL Environment.
    
    This Multi-Layer Perceptron (MLP) maps the Phase 2 50-dimensional game
    observation vector to 70 raw action logits representing action preferences.
    """
    def __init__(self, state_dim: int = OBSERVATION_DIM, action_dim: int = ACTION_DIM):
        super(ActorNetwork, self).__init__()
        if state_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"Phase 2 ActorNetwork requires state_dim={OBSERVATION_DIM} and "
                f"action_dim={ACTION_DIM}; got state_dim={state_dim}, action_dim={action_dim}."
            )
        
        # Layer 1: Input to hidden mapping
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Feature processing
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Raw output logits projection
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Actor network to compute raw action logits."""
        # Enforce float32 precision
        x = x.to(dtype=torch.float32)
        if x.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 ActorNetwork expects final input dimension {OBSERVATION_DIM}, "
                f"got {x.shape[-1]}."
            )
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        logits = self.fc3(out)
        return logits


class CriticNetwork(nn.Module):
    """PPO Critic Network (Value Function) for the Mini Pokémon TCG RL Environment.
    
    This Multi-Layer Perceptron (MLP) maps the Phase 2 50-dimensional game
    observation vector to a single scalar value V(s) representing the expected return.
    """
    def __init__(self, state_dim: int = OBSERVATION_DIM):
        super(CriticNetwork, self).__init__()
        if state_dim != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 CriticNetwork requires state_dim={OBSERVATION_DIM}; got state_dim={state_dim}."
            )
        
        # Layer 1: Input to hidden mapping
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Feature processing
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Scalar value projection V(s)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Critic network to compute state value estimations V(s)."""
        x = x.to(dtype=torch.float32)
        if x.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 CriticNetwork expects final input dimension {OBSERVATION_DIM}, "
                f"got {x.shape[-1]}."
            )
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        state_value = self.fc3(out)
        return state_value


class MaskedCategorical:
    """Helper utility for action selection in PPO using action masking.
    
    Wraps PyTorch's standard Categorical distribution for the Phase 2 70-action
    interface. Illegal logits are replaced before probability calculation, so
    illegal actions have zero probability. A binary mask with at least one legal
    action is required for every state; this includes forced-promotion states
    whose only legal choices can be actions 68 and/or 69.
    """
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        """Initializes the Masked Categorical distribution.
        
        Args:
            logits: Tensor shaped (70,) or (B, 70) containing raw action preferences.
            mask: Binary tensor shaped (70,) or (B, 70), with 1 for legal and 0 for illegal actions.
        """
        if logits.ndim not in (1, 2):
            raise ValueError(f"Policy logits must have rank 1 or 2, got shape {tuple(logits.shape)}.")
        if logits.shape != mask.shape:
            raise ValueError(
                f"Policy logits and action mask must have matching shapes, got "
                f"{tuple(logits.shape)} and {tuple(mask.shape)}."
            )
        if logits.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"Phase 2 PPO policy expects {ACTION_DIM} logits, got {logits.shape[-1]}."
            )
        if logits.device != mask.device:
            raise ValueError("Policy logits and action mask must be on the same device.")
        if not torch.isfinite(logits).all():
            raise ValueError("Policy logits must be finite.")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Action mask must be binary.")

        self.original_shape = logits.shape
        
        # Enforce batch layout internally for robust tensor alignment
        self.is_batched = logits.ndim == 2
        
        if not self.is_batched:
            logits_batched = logits.unsqueeze(0)
            mask_batched = mask.unsqueeze(0)
        else:
            logits_batched = logits
            mask_batched = mask

        if torch.any(mask_batched.sum(dim=-1) == 0):
            raise ValueError("Action mask has no legal actions for at least one state.")

        # Use the most-negative finite value for the tensor dtype. With at least one
        # legal action per row, this keeps one-action forced-promotion distributions
        # finite while assigning exactly zero sampling probability to illegal actions.
        illegal_logit = torch.finfo(logits_batched.dtype).min
        masked_logits = logits_batched.masked_fill(mask_batched == 0, illegal_logit)
        
        self.dist = Categorical(logits=masked_logits)
        self.mask_batched = mask_batched

    def sample(self) -> torch.Tensor:
        """Samples action indices from the distribution. Guaranteed to be legal."""
        samples = self.dist.sample()
        if not self.is_batched:
            return samples.squeeze(0)
        return samples

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Computes the log probability for the selected actions."""
        # Categorical expects matching batch alignments
        if not self.is_batched and len(action.shape) == 0:
            action = action.unsqueeze(0)
            log_p = self.dist.log_prob(action)
            return log_p.squeeze(0)
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Computes the Shannon entropy of the legal probability distribution."""
        ent = self.dist.entropy()
        if not self.is_batched:
            return ent.squeeze(0)
        return ent

    @property
    def probs(self) -> torch.Tensor:
        """Returns the calculated probability distribution over actions."""
        p = self.dist.probs
        if not self.is_batched:
            return p.squeeze(0)
        return p


class PPORolloutBuffer:
    """Rollout Buffer for storing transitions gathered during PPO environment interaction.
    
    Saves state vectors, chosen action indices, rewards, termination signals, value estimations, 
    action masks, and action log-probabilities. Supports clean clearing and vectorization to device.
    """
    def __init__(self):
        self.clear()

    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        done: bool, 
        value: float, 
        log_prob: float, 
        action_mask: np.ndarray
    ) -> None:
        """Saves a single environment step transition."""
        state = np.asarray(state, dtype=np.float32)
        action_mask = np.asarray(action_mask, dtype=np.float32)
        if state.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"PPORolloutBuffer expects Phase 2 states shaped ({OBSERVATION_DIM},), got {state.shape}."
            )
        if action_mask.shape != (ACTION_DIM,):
            raise ValueError(
                f"PPORolloutBuffer expects Phase 2 action masks shaped ({ACTION_DIM},), "
                f"got {action_mask.shape}."
            )
        if not np.all((action_mask == 0.0) | (action_mask == 1.0)):
            raise ValueError("Action mask must be binary.")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < ACTION_DIM:
            raise ValueError(f"Action must be in [0, {ACTION_DIM - 1}].")
        if action_mask[int(action)] != 1.0:
            raise ValueError(f"Cannot store illegal action {action} in a PPO rollout.")

        self.states.append(state.copy())
        self.actions.append(int(action))
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask.copy())

    def clear(self) -> None:
        """Flushes the buffer. Typically executed after completing policy optimization updates."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.action_masks: List[np.ndarray] = []

    def length(self) -> int:
        """Returns the current number of experiences stored inside the buffer."""
        return len(self.states)

    def __len__(self) -> int:
        return self.length()

    def get_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts all stored lists to PyTorch Tensors on the designated device.
        
        Args:
            device: Active target hardware device (CPU or CUDA GPU).
            
        Returns:
            Tensors: (states, actions, rewards, dones, values, log_probs, action_masks)
        """
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        rewards_t = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        dones_t = torch.tensor(np.array(self.dones), dtype=torch.float32, device=device)
        values_t = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)
        log_probs_t = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        action_masks_t = torch.tensor(np.array(self.action_masks), dtype=torch.float32, device=device)
        
        return states_t, actions_t, rewards_t, dones_t, values_t, log_probs_t, action_masks_t
