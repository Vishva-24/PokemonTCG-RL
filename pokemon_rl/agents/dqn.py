import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, List, Optional

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

class DQNetwork(nn.Module):
    """Deep Q-Network (MLP) for Mini Pokémon TCG RL Environment.
    
    This simple Multi-Layer Perceptron (MLP) maps the Phase 2 50-dimensional
    game observation vector to 70 Q-values, one for each possible action.
    """
    def __init__(self, state_dim: int = OBSERVATION_DIM, action_dim: int = ACTION_DIM):
        super(DQNetwork, self).__init__()
        if state_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"Phase 2 DQNetwork requires state_dim={OBSERVATION_DIM} and action_dim={ACTION_DIM}; "
                f"got state_dim={state_dim}, action_dim={action_dim}."
            )
        
        # Layer 1: Input to Hidden Representation
        # Transforms the 50-element state vector into a higher-dimensional 128-element feature space.
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Feature Processing
        # Processes the non-linear features of the game state to find complex associations.
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Output Projection
        # Maps the 128-dimensional hidden representation to the 70-dimensional action space.
        # Outputs raw, unnormalized Q-values representing expected future cumulative returns.
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the DQN network.
        
        Supports both single state vectors (shape: [50] or [1, 50]) and batched state vectors (shape: [B, 50]).
        """
        # Ensure correct floating point precision
        x = x.to(dtype=torch.float32)
        if x.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 DQNetwork expects final input dimension {OBSERVATION_DIM}, got {x.shape[-1]}."
            )
        
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        q_values = self.fc3(out)
        return q_values


class ReplayBuffer:
    """Standard experience replay buffer for off-policy DQN learning.
    
    Stores transitions (state, action, reward, next_state, done) and returns
    random minibatches as PyTorch tensors for model optimization.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Stores a single transition experience inside the buffer.
        
        Operates as a cyclic/circular array to maintain constant capacity.
        """
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (OBSERVATION_DIM,) or next_state.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"ReplayBuffer expects Phase 2 states shaped ({OBSERVATION_DIM},), "
                f"got {state.shape} and {next_state.shape}."
            )
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < ACTION_DIM:
            raise ValueError(f"ReplayBuffer action must be in [0, {ACTION_DIM - 1}].")
        transition = (state, int(action), reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a random minibatch from experience.
        
        Converts numpy arrays to PyTorch float/long tensors for neural net training.
        """
        minibatch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(
    state: np.ndarray, 
    model: nn.Module, 
    action_mask: np.ndarray, 
    epsilon: float, 
    device: torch.device = torch.device("cpu")
) -> int:
    """Selects an action using epsilon-greedy behavior with strict action masking.
    
    - With probability epsilon: chooses a random index among legal actions (mask == 1.0).
    - Otherwise: chooses the index of the highest-Q legal action, masking illegal actions to -1e9.
    
    Guarantees that an illegal action is never selected.
    """
    state = np.asarray(state, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=np.float32)
    if state.shape != (OBSERVATION_DIM,):
        raise ValueError(f"DQN expects a Phase 2 state shaped ({OBSERVATION_DIM},), got {state.shape}.")
    if action_mask.shape != (ACTION_DIM,):
        raise ValueError(f"DQN expects a Phase 2 action mask shaped ({ACTION_DIM},), got {action_mask.shape}.")
    if not np.all((action_mask == 0.0) | (action_mask == 1.0)):
        raise ValueError("Action mask must be binary.")

    # 1. Identify legal action indices from the 70-action Phase 2 mask.
    legal_indices = np.where(action_mask == 1.0)[0]
    if len(legal_indices) == 0:
        raise ValueError("Action mask has no legal actions! Game state is blocked.")
        
    # 2. Explore: select a random legal action with probability epsilon
    if random.random() < epsilon:
        return int(random.choice(legal_indices))
        
    # 3. Exploit: select the best action using masked Q-values
    model.eval()
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # [1, 50]
        q_values_t = model(state_t).squeeze(0) # [70]
        q_values = q_values_t.cpu().numpy()
        if q_values.shape != (ACTION_DIM,):
            raise ValueError(
                f"DQN model returned {q_values.shape}; expected ({ACTION_DIM},) Q-values for the Phase 2 action space."
            )
        
        # Apply action masking by setting illegal actions to a very large negative value
        masked_q_values = np.copy(q_values)
        masked_q_values[action_mask == 0.0] = -1e9
        
        # Return index of maximum Q-value among legal options
        best_action = np.argmax(masked_q_values)
        
        # Double-check safety to guarantee we didn't pick a masked action
        if action_mask[best_action] == 0.0:
            # Fallback to choosing any random legal index if masking logic collapsed
            return int(random.choice(legal_indices))
            
        return int(best_action)
