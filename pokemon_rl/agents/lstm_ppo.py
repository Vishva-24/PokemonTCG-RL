import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Optional

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

class LSTMActorNetwork(nn.Module):
    """PPO Recurrent Actor Network (Policy) for the Mini Pokémon TCG RL Environment.
    
    Processes the Phase 2 50-dimensional state space with spatial Dense layers and
    temporal LSTM memory before mapping to the 70-element action-logit space.
    """
    def __init__(self, state_dim: int = OBSERVATION_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 128):
        super(LSTMActorNetwork, self).__init__()
        if state_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"Phase 2 LSTMActorNetwork requires state_dim={OBSERVATION_DIM} and "
                f"action_dim={ACTION_DIM}; got state_dim={state_dim}, action_dim={action_dim}."
            )
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes zeros for hidden and cell states of shape (1, B, 128)."""
        h0 = torch.zeros(1, batch_size, self.hidden_dim, dtype=torch.float32, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, dtype=torch.float32, device=device)
        return (h0, c0)

    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the LSTM Actor network.
        
        Supports both single-step inputs (shape: (50,) or (1, 1, 50)) and batched
        sequence inputs (shape: (B, L, 50)).
        If dones is provided, hidden states are zeroed out at episode boundaries step-by-step.
        """
        x = x.to(dtype=torch.float32)
        original_ndim = x.ndim
        if original_ndim not in (1, 2, 3):
            raise ValueError(f"Actor input must have rank 1, 2, or 3, got shape {tuple(x.shape)}.")
        if x.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 LSTMActorNetwork expects final input dimension {OBSERVATION_DIM}, "
                f"got {x.shape[-1]}."
            )
        
        # Enforce sequence layout internally for nn.LSTM (B, L, feature)
        if original_ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_ndim == 2:
            x = x.unsqueeze(1)
            
        B, L, _ = x.shape
        
        if hidden is None:
            hidden = self.init_hidden(B, x.device)

        if dones is not None and dones.shape != (B, L):
            raise ValueError(
                f"Actor done flags must have shape ({B}, {L}), got {tuple(dones.shape)}."
            )
            
        features = self.relu(self.fc1(x))
        
        if dones is not None and L > 1:
            # Step-by-step done reset logic to prevent temporal leakage inside sequence minibatches
            lstm_outs = []
            curr_h, curr_c = hidden
            for t in range(L):
                if t > 0:
                    done_prev = dones[:, t-1].view(1, -1, 1) # Reshaped to (1, B, 1)
                    curr_h = curr_h * (1.0 - done_prev)
                    curr_c = curr_c * (1.0 - done_prev)
                
                feat_step = features[:, t:t+1, :] # shape (B, 1, hidden_dim)
                out_step, (curr_h, curr_c) = self.lstm(feat_step, (curr_h, curr_c))
                lstm_outs.append(out_step)
                
            lstm_out = torch.cat(lstm_outs, dim=1)
            next_hidden = (curr_h, curr_c)
        else:
            lstm_out, next_hidden = self.lstm(features, hidden)
            
        logits = self.fc2(lstm_out)
        
        # Revert squeeze structure for conversational execution compatibility
        if original_ndim == 1:
            logits = logits.squeeze(0).squeeze(0)
        elif original_ndim == 2:
            logits = logits.squeeze(1)
            
        return logits, next_hidden


class LSTMCriticNetwork(nn.Module):
    """PPO Recurrent Critic Network (Value Function) for the Mini Pokémon TCG RL Environment.
    
    Processes state spaces with an LSTM architecture to output expected value forecasts V(s).
    """
    def __init__(self, state_dim: int = OBSERVATION_DIM, hidden_dim: int = 128):
        super(LSTMCriticNetwork, self).__init__()
        if state_dim != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 LSTMCriticNetwork requires state_dim={OBSERVATION_DIM}; got state_dim={state_dim}."
            )
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes zeros for hidden and cell states of shape (1, B, 128)."""
        h0 = torch.zeros(1, batch_size, self.hidden_dim, dtype=torch.float32, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, dtype=torch.float32, device=device)
        return (h0, c0)

    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the LSTM Critic network.
        
        Supports both single-step inputs (shape: (50,) or (1, 1, 50)) and batched
        sequence inputs (shape: (B, L, 50)).
        If dones is provided, hidden states are zeroed out at episode boundaries step-by-step.
        """
        x = x.to(dtype=torch.float32)
        original_ndim = x.ndim
        if original_ndim not in (1, 2, 3):
            raise ValueError(f"Critic input must have rank 1, 2, or 3, got shape {tuple(x.shape)}.")
        if x.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 LSTMCriticNetwork expects final input dimension {OBSERVATION_DIM}, "
                f"got {x.shape[-1]}."
            )
        
        if original_ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_ndim == 2:
            x = x.unsqueeze(1)
            
        B, L, _ = x.shape
        
        if hidden is None:
            hidden = self.init_hidden(B, x.device)

        if dones is not None and dones.shape != (B, L):
            raise ValueError(
                f"Critic done flags must have shape ({B}, {L}), got {tuple(dones.shape)}."
            )
            
        features = self.relu(self.fc1(x))
        
        if dones is not None and L > 1:
            # Step-by-step done reset logic to prevent temporal leakage inside sequence minibatches
            lstm_outs = []
            curr_h, curr_c = hidden
            for t in range(L):
                if t > 0:
                    done_prev = dones[:, t-1].view(1, -1, 1) # Reshaped to (1, B, 1)
                    curr_h = curr_h * (1.0 - done_prev)
                    curr_c = curr_c * (1.0 - done_prev)
                
                feat_step = features[:, t:t+1, :] # shape (B, 1, hidden_dim)
                out_step, (curr_h, curr_c) = self.lstm(feat_step, (curr_h, curr_c))
                lstm_outs.append(out_step)
                
            lstm_out = torch.cat(lstm_outs, dim=1)
            next_hidden = (curr_h, curr_c)
        else:
            lstm_out, next_hidden = self.lstm(features, hidden)
            
        values = self.fc2(lstm_out)
        
        if original_ndim == 1:
            values = values.squeeze(0).squeeze(0)
        elif original_ndim == 2:
            values = values.squeeze(1)
            
        return values, next_hidden


class MaskedCategorical:
    """Recurrent-compatible Phase 2 masked policy distribution for PPO sequences."""
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        """Initializes MaskedCategorical distribution.
        
        Supports layouts (70,), (B, 70), or (B, L, 70). Every mask row must
        be binary and contain at least one legal action, including forced-
        promotion rows where only action 68 and/or 69 is legal.
        """
        if logits.ndim not in (1, 2, 3):
            raise ValueError(f"Policy logits must have rank 1, 2, or 3, got shape {tuple(logits.shape)}.")
        if logits.shape != mask.shape:
            raise ValueError(
                f"Policy logits and action mask must have matching shapes, got "
                f"{tuple(logits.shape)} and {tuple(mask.shape)}."
            )
        if logits.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"Phase 2 recurrent PPO policy expects {ACTION_DIM} logits, got {logits.shape[-1]}."
            )
        if logits.device != mask.device:
            raise ValueError("Policy logits and action mask must be on the same device.")
        if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
            raise ValueError("Policy logits must be finite floating-point values.")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Action mask must be binary.")
        if torch.any(mask.sum(dim=-1) == 0):
            raise ValueError("Action mask has no legal actions for at least one state.")

        self.original_shape = logits.shape
        # A finite lower bound preserves stable categorical probabilities even
        # when a forced-promotion state has exactly one legal action.
        masked_logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        
        self.dist = Categorical(logits=masked_logits)
        self.mask = mask

    def sample(self) -> torch.Tensor:
        """Samples legal actions. Guaranteed to exclude illegal options."""
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Calculates action log probabilities."""
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Calculates Shannon policy distribution entropy."""
        return self.dist.entropy()

    @property
    def probs(self) -> torch.Tensor:
        """Returns action probabilities."""
        return self.dist.probs


class LSTMPPORolloutBuffer:
    """Rollout Buffer for recurrent PPO, storing transitions and initial hidden state checkpoints."""
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
        action_mask: np.ndarray,
        actor_h: torch.Tensor,
        actor_c: torch.Tensor,
        critic_h: torch.Tensor,
        critic_c: torch.Tensor
    ):
        """Appends a transition step. Memory coordinates are detached onto CPU storage."""
        state = np.asarray(state, dtype=np.float32)
        action_mask = np.asarray(action_mask, dtype=np.float32)
        if state.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"LSTMPPORolloutBuffer expects Phase 2 states shaped ({OBSERVATION_DIM},), "
                f"got {state.shape}."
            )
        if action_mask.shape != (ACTION_DIM,):
            raise ValueError(
                f"LSTMPPORolloutBuffer expects Phase 2 action masks shaped ({ACTION_DIM},), "
                f"got {action_mask.shape}."
            )
        if not np.all((action_mask == 0.0) | (action_mask == 1.0)):
            raise ValueError("Action mask must be binary.")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < ACTION_DIM:
            raise ValueError(f"Action must be in [0, {ACTION_DIM - 1}].")
        if action_mask[int(action)] != 1.0:
            raise ValueError(f"Cannot store illegal action {action} in a recurrent PPO rollout.")
        for name, h, c in (
            ("actor", actor_h, actor_c),
            ("critic", critic_h, critic_c),
        ):
            if h.ndim != 3 or c.ndim != 3 or h.shape != c.shape or h.shape[:2] != (1, 1):
                raise ValueError(
                    f"Stored {name} hidden and cell states must have matching shape (1, 1, H); "
                    f"got {tuple(h.shape)} and {tuple(c.shape)}."
                )

        self.states.append(state.copy())
        self.actions.append(int(action))
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask.copy())
        self.actor_hs.append(actor_h.detach().cpu().clone())
        self.actor_cs.append(actor_c.detach().cpu().clone())
        self.critic_hs.append(critic_h.detach().cpu().clone())
        self.critic_cs.append(critic_c.detach().cpu().clone())

    def clear(self):
        """Flushes rollout experience storage lists."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.action_masks: List[np.ndarray] = []
        self.actor_hs: List[torch.Tensor] = []
        self.actor_cs: List[torch.Tensor] = []
        self.critic_hs: List[torch.Tensor] = []
        self.critic_cs: List[torch.Tensor] = []

    def length(self) -> int:
        """Returns stored transition count."""
        return len(self.states)

    def __len__(self) -> int:
        return self.length()

    def get_sequence_batches(
        self, 
        sequence_length: int = 16, 
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reformats flat transitions list into batched sequential tensors for recurrent backprop.
        
        Args:
            sequence_length: Temporal length L (default=16)
            device: Active device
            
        Returns:
            Tuple of Tensors: (states, action_masks, actions, rewards, dones, values, log_probs, actor_h0, actor_c0, critic_h0, critic_c0)
        """
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")
        T = self.length()
        num_sequences = T // sequence_length
        if num_sequences == 0:
            raise ValueError(
                f"Need at least {sequence_length} transitions to form a recurrent PPO sequence; got {T}."
            )
        truncated_len = num_sequences * sequence_length
        
        states_t = torch.tensor(np.array(self.states[:truncated_len]), dtype=torch.float32, device=device)
        states_seq = states_t.view(num_sequences, sequence_length, OBSERVATION_DIM)
        
        action_masks_t = torch.tensor(np.array(self.action_masks[:truncated_len]), dtype=torch.float32, device=device)
        action_masks_seq = action_masks_t.view(num_sequences, sequence_length, ACTION_DIM)
        
        actions_t = torch.tensor(np.array(self.actions[:truncated_len]), dtype=torch.long, device=device)
        actions_seq = actions_t.view(num_sequences, sequence_length)
        
        rewards_t = torch.tensor(np.array(self.rewards[:truncated_len]), dtype=torch.float32, device=device)
        rewards_seq = rewards_t.view(num_sequences, sequence_length)
        
        dones_t = torch.tensor(np.array(self.dones[:truncated_len]), dtype=torch.float32, device=device)
        dones_seq = dones_t.view(num_sequences, sequence_length)
        
        values_t = torch.tensor(np.array(self.values[:truncated_len]), dtype=torch.float32, device=device)
        values_seq = values_t.view(num_sequences, sequence_length)
        
        log_probs_t = torch.tensor(np.array(self.log_probs[:truncated_len]), dtype=torch.float32, device=device)
        log_probs_seq = log_probs_t.view(num_sequences, sequence_length)
        
        # Grab initial Actor and Critic hidden/cell state variables at start boundaries of each sequence block
        actor_h_list = [self.actor_hs[i * sequence_length] for i in range(num_sequences)]
        actor_c_list = [self.actor_cs[i * sequence_length] for i in range(num_sequences)]
        critic_h_list = [self.critic_hs[i * sequence_length] for i in range(num_sequences)]
        critic_c_list = [self.critic_cs[i * sequence_length] for i in range(num_sequences)]
        
        # Concatenate along batch axis dim 1 to get shape (1, B, 128)
        actor_h_seq = torch.cat(actor_h_list, dim=1).to(device)
        actor_c_seq = torch.cat(actor_c_list, dim=1).to(device)
        critic_h_seq = torch.cat(critic_h_list, dim=1).to(device)
        critic_c_seq = torch.cat(critic_c_list, dim=1).to(device)
        
        return (states_seq, action_masks_seq, actions_seq, rewards_seq, dones_seq, values_seq, log_probs_seq,
                actor_h_seq, actor_c_seq, critic_h_seq, critic_c_seq)
