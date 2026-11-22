import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple

from pokemon_rl.agents.dqn import DQNetwork, ReplayBuffer
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def compute_batch_action_masks(states: torch.Tensor) -> torch.Tensor:
    """Reconstruct the Phase 2 70-action legal masks from 50-D observations.
    
    Args:
        states: PyTorch Tensor of shape (B, 50)
        
    Returns:
        PyTorch Tensor of shape (B, 70) with 1.0 for legal and 0.0 for illegal actions.

    The replay buffer intentionally keeps its existing transition format, so
    target-Q calculation reconstructs a next-state mask from the information
    set. This exactly follows the locked Phase 2 index layout and includes the
    forced-promotion rule: when index 29 is set, only actions 68 and/or 69 may
    be legal.
    """
    if states.ndim != 2 or states.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"Expected batched Phase 2 states shaped (B, {OBSERVATION_DIM}), got {tuple(states.shape)}.")

    B = states.shape[0]
    masks = torch.zeros((B, ACTION_DIM), dtype=torch.float32, device=states.device)

    active_hp, active_type = states[:, 0], states[:, 5]
    active_grass = states[:, 1]
    active_fire = states[:, 2]
    active_water = states[:, 3]
    active_lightning = states[:, 4]
    total_active_energy = states[:, 1:5].sum(dim=1)
    bench1_hp, bench1_type = states[:, 6], states[:, 11]
    bench2_hp, bench2_type = states[:, 12], states[:, 17]
    hand = states[:, 18:26]
    energy_attached_this_turn = states[:, 27] > 0.0
    retreated_this_turn = states[:, 28] > 0.0
    forced_promotion = states[:, 29] > 0.0

    active_exists = active_type > 0.0
    bench1_exists = bench1_type > 0.0
    bench2_exists = bench2_type > 0.0
    bench_exists = bench1_exists | bench2_exists
    normal_turn = ~forced_promotion

    # 0 End Turn. In a forced-promotion state no normal action is legal.
    masks[:, 0] = normal_turn.float()

    # 1 Attack 1: Grass=Applin, Fire=Charmander, Water=Swinub,
    # Lightning=Pikachu. All four have an Attack 1 with the costs below.
    attack1_legal = (
        ((active_type == 1.0) & (active_grass >= 1.0))
        | ((active_type == 2.0) & (active_fire >= 1.0) & (total_active_energy >= 2.0))
        | ((active_type == 3.0) & (total_active_energy >= 1.0))
        | ((active_type == 4.0) & (active_lightning >= 1.0))
    )
    masks[:, 1] = (normal_turn & attack1_legal).float()

    # 2 Attack 2 exists only for Swinub and Pikachu.
    attack2_legal = (
        ((active_type == 3.0) & (active_water >= 1.0) & (total_active_energy >= 2.0))
        | ((active_type == 4.0) & (active_lightning >= 1.0) & (total_active_energy >= 2.0))
    )
    masks[:, 2] = (normal_turn & attack2_legal).float()

    # 3 Retreat. Required costs are 1 for Applin/Charmander/Pikachu and 2 for Swinub.
    retreat_cost_met = (
        ((active_type == 1.0) & (total_active_energy >= 1.0))
        | ((active_type == 2.0) & (total_active_energy >= 1.0))
        | ((active_type == 3.0) & (total_active_energy >= 2.0))
        | ((active_type == 4.0) & (total_active_energy >= 1.0))
    )
    masks[:, 3] = (normal_turn & active_exists & bench_exists & ~retreated_this_turn & retreat_cost_met).float()

    # Hand IDs: 1-4 Pokemon, 5-8 typed Energy, 9 Potion, 10 compatibility-only Colorless Energy.
    for hand_slot in range(8):
        card_type = hand[:, hand_slot]
        is_pokemon = (card_type >= 1.0) & (card_type <= 4.0)
        is_energy = ((card_type >= 5.0) & (card_type <= 8.0)) | (card_type == 10.0)
        is_potion = card_type == 9.0

        # 4-19: Play a Basic Pokemon to an empty Bench slot.
        masks[:, 4 + hand_slot] = (normal_turn & is_pokemon & ~bench1_exists).float()
        masks[:, 12 + hand_slot] = (normal_turn & is_pokemon & ~bench2_exists).float()

        # 20-43: One manual Energy attachment per turn, to an occupied target.
        can_attach = normal_turn & is_energy & ~energy_attached_this_turn
        masks[:, 20 + hand_slot] = (can_attach & active_exists).float()
        masks[:, 28 + hand_slot] = (can_attach & bench1_exists).float()
        masks[:, 36 + hand_slot] = (can_attach & bench2_exists).float()

        # 44-67: Potion can target exactly a damaged in-play Pokemon.
        active_damaged = (
            ((active_type == 1.0) & (active_hp < 40.0))
            | ((active_type == 2.0) & (active_hp < 80.0))
            | ((active_type == 3.0) & (active_hp < 70.0))
            | ((active_type == 4.0) & (active_hp < 70.0))
        )
        bench1_damaged = (
            ((bench1_type == 1.0) & (bench1_hp < 40.0))
            | ((bench1_type == 2.0) & (bench1_hp < 80.0))
            | ((bench1_type == 3.0) & (bench1_hp < 70.0))
            | ((bench1_type == 4.0) & (bench1_hp < 70.0))
        )
        bench2_damaged = (
            ((bench2_type == 1.0) & (bench2_hp < 40.0))
            | ((bench2_type == 2.0) & (bench2_hp < 80.0))
            | ((bench2_type == 3.0) & (bench2_hp < 70.0))
            | ((bench2_type == 4.0) & (bench2_hp < 70.0))
        )
        masks[:, 44 + hand_slot] = (normal_turn & is_potion & active_damaged).float()
        masks[:, 52 + hand_slot] = (normal_turn & is_potion & bench1_damaged).float()
        masks[:, 60 + hand_slot] = (normal_turn & is_potion & bench2_damaged).float()

    # 68-69: Forced promotion is the only legal action family when pending.
    masks[:, 68] = (forced_promotion & bench1_exists).float()
    masks[:, 69] = (forced_promotion & bench2_exists).float()
    return masks


class DQNTrainer:
    """DQN Trainer managing networks, target sync, buffer sampling, and gradient updates."""
    def __init__(
        self,
        state_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        lr: float = 1e-4,
        gamma: float = 0.99,
        capacity: int = 10000,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu")
    ):
        if state_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"Phase 2 DQN requires state_dim={OBSERVATION_DIM} and action_dim={ACTION_DIM}; "
                f"got state_dim={state_dim}, action_dim={action_dim}."
            )
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        
        # 1. Online policy network
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        
        # 2. Target network
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        
        # Synchronize networks initially
        self.update_target_network()
        self.target_net.eval()  # Target net is strictly evaluation mode
        
        # 3. Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity)
        
        # 4. Optimizer & Huber Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss
        
    def update_target_network(self) -> None:
        """Copies weights from the online policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def train_step(self) -> Optional[float]:
        """Performs a single gradient descent update step over a sampled minibatch.
        
        Returns:
            The scalar loss value (Huber loss) as a float, or None if buffer lacks enough transitions.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # 1. Sample minibatch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move inputs to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 2. Compute current Q-values for actions taken: Q(s, a)
        q_values = self.policy_net(states)
        current_q = q_values.gather(1, actions)
        
        # 3. Compute target Q-values strictly masking out illegal actions
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            
            # Action mask for next state
            next_masks = compute_batch_action_masks(next_states)
            
            # Set Q-values of illegal next-state actions to -1e9
            masked_next_q_values = next_q_values.clone()
            masked_next_q_values[next_masks == 0.0] = -1e9
            
            # Select max Q-value among legal actions
            max_next_q, _ = torch.max(masked_next_q_values, dim=1, keepdim=True)
            
            # Compute TD Target: y = reward + (1 - done) * gamma * max_legal_Q(next_state)
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q
            
        # 4. Compute Loss
        loss = self.loss_fn(current_q, target_q)
        
        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return float(loss.item())
