import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, List

from pokemon_rl.agents.lstm_ppo import LSTMActorNetwork, LSTMCriticNetwork, MaskedCategorical
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def compute_recurrent_gae_and_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Generalized Advantage Estimation (GAE) and Discounted Target Returns."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            
        # TD Residual
        delta = rewards[t] + gamma * next_non_terminal * next_value - values[t]
        
        # Recurrent GAE
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
    returns = advantages + values
    
    if normalize_advantages and T > 1:
        std = advantages.std()
        mean = advantages.mean()
        advantages = (advantages - mean) / (std + 1e-8)
        
    return advantages, returns


class LSTMPPOTrainer:
    """Trainer orchestrator for Recurrent LSTM-PPO, managing recurrent backpropagation through time (BPTT)."""
    def __init__(
        self,
        actor: LSTMActorNetwork,
        critic: LSTMCriticNetwork,
        lr: float = 5e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        sequence_length: int = 16,
        batch_size: int = 4,
        device: torch.device = torch.device("cpu")
    ):
        self.actor = actor
        self.critic = critic
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coefficient
        self.vf_coef = value_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        
        # Separate optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.loss_fn = nn.MSELoss()

    def _validate_update_inputs(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        actor_h0: torch.Tensor,
        actor_c0: torch.Tensor,
        critic_h0: torch.Tensor,
        critic_c0: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Validate fixed Phase 2 recurrent PPO sequence tensors before BPTT."""
        if states.ndim != 3 or states.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"LSTM-PPO states must have shape (B, L, {OBSERVATION_DIM}), got {tuple(states.shape)}."
            )
        batch_count, sequence_length, _ = states.shape
        if batch_count == 0:
            raise ValueError("LSTM-PPO cannot update from an empty sequence batch.")
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"LSTM-PPO expected sequence length {self.sequence_length}, got {sequence_length}."
            )
        expected_step_shape = (batch_count, sequence_length)
        if action_masks.shape != (*expected_step_shape, ACTION_DIM):
            raise ValueError(
                f"LSTM-PPO action masks must have shape (B, L, {ACTION_DIM}), "
                f"got {tuple(action_masks.shape)}."
            )
        if actions.shape != expected_step_shape:
            raise ValueError(
                f"LSTM-PPO actions must have shape {expected_step_shape}, got {tuple(actions.shape)}."
            )
        if actions.dtype != torch.long:
            raise ValueError("LSTM-PPO actions must be torch.long action indices.")
        for name, tensor in {
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
            "dones": dones,
        }.items():
            if tensor.shape != expected_step_shape:
                raise ValueError(
                    f"LSTM-PPO {name} must have shape {expected_step_shape}, got {tuple(tensor.shape)}."
                )
        if not torch.all((action_masks == 0) | (action_masks == 1)):
            raise ValueError("LSTM-PPO action masks must be binary.")
        if torch.any(action_masks.sum(dim=-1) == 0):
            raise ValueError("Every LSTM-PPO action mask must contain at least one legal action.")
        if not torch.all((dones == 0) | (dones == 1)):
            raise ValueError("LSTM-PPO done flags must be binary.")
        if torch.any(actions < 0) or torch.any(actions >= ACTION_DIM):
            raise ValueError(f"LSTM-PPO actions must be in [0, {ACTION_DIM - 1}].")
        if torch.any(torch.gather(action_masks, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1) != 1):
            raise ValueError("LSTM-PPO rollout contains an action that is illegal under its stored mask.")

        expected_actor_hidden_shape = (1, batch_count, self.actor.hidden_dim)
        expected_critic_hidden_shape = (1, batch_count, self.critic.hidden_dim)
        for name, tensor, expected_shape in (
            ("actor_h0", actor_h0, expected_actor_hidden_shape),
            ("actor_c0", actor_c0, expected_actor_hidden_shape),
            ("critic_h0", critic_h0, expected_critic_hidden_shape),
            ("critic_c0", critic_c0, expected_critic_hidden_shape),
        ):
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"LSTM-PPO {name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
                )

    def update(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        actor_h0: torch.Tensor,
        actor_c0: torch.Tensor,
        critic_h0: torch.Tensor,
        critic_c0: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, Any]:
        """Performs PPO updates using sequence batches with step-by-step done reset logic."""
        states = states.to(self.device)
        action_masks = action_masks.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device).detach()
        returns = returns.to(self.device).detach()
        advantages = advantages.to(self.device).detach()
        dones = dones.to(self.device)
        
        # Detach initial recurrent states
        actor_h0 = actor_h0.to(self.device).detach()
        actor_c0 = actor_c0.to(self.device).detach()
        critic_h0 = critic_h0.to(self.device).detach()
        critic_c0 = critic_c0.to(self.device).detach()

        self._validate_update_inputs(
            states, action_masks, actions, old_log_probs, returns, advantages,
            actor_h0, actor_c0, critic_h0, critic_c0, dones
        )

        B_total = states.shape[0]
        
        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        entropy_accum = 0.0
        kl_accum = 0.0
        clip_fraction_accum = 0.0
        update_count = 0

        for epoch in range(self.ppo_epochs):
            indices = np.arange(B_total)
            np.random.shuffle(indices)

            for start_idx in range(0, B_total, self.batch_size):
                end_idx = min(start_idx + self.batch_size, B_total)
                batch_indices = indices[start_idx:end_idx]
                curr_batch_size = len(batch_indices)

                b_states = states[batch_indices]
                b_action_masks = action_masks[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]
                b_dones = dones[batch_indices]
                
                # Slice and detach initial hidden states along the batch dimension (dim 1)
                b_actor_h0 = actor_h0[:, batch_indices, :].detach()
                b_actor_c0 = actor_c0[:, batch_indices, :].detach()
                b_critic_h0 = critic_h0[:, batch_indices, :].detach()
                b_critic_c0 = critic_c0[:, batch_indices, :].detach()

                # 1. Forward pass recurrent Actor with b_dones
                b_logits, _ = self.actor(b_states, (b_actor_h0, b_actor_c0), dones=b_dones)

                # 2. Mask logits and generate policy distribution
                b_policy = MaskedCategorical(b_logits, b_action_masks)

                # 3. Compute log probabilities
                b_new_log_probs = b_policy.log_prob(b_actions)

                # 4. Compute PPO ratio
                b_ratios = torch.exp(b_new_log_probs - b_old_log_probs)

                # 5. Clipped objective
                surr1 = b_ratios * b_advantages
                surr2 = torch.clamp(b_ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                b_l_clip = torch.min(surr1, surr2)
                policy_loss = -torch.mean(b_l_clip)

                # 6. Forward pass recurrent Critic with b_dones
                b_values, _ = self.critic(b_states, (b_critic_h0, b_critic_c0), dones=b_dones)
                b_new_values = b_values.squeeze(-1)
                value_loss = self.loss_fn(b_new_values, b_returns)

                # 7. Entropy
                entropy = torch.mean(b_policy.entropy())

                # Combined loss
                total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                # 8. Optimize Actor
                self.actor_optimizer.zero_grad()
                actor_loss = policy_loss - self.entropy_coef * entropy
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 9. Optimize Critic
                self.critic_optimizer.zero_grad()
                critic_loss = self.vf_coef * value_loss
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 10. Track metrics
                with torch.no_grad():
                    log_ratio = b_new_log_probs - b_old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).item()
                    
                    clipped = (b_ratios < (1.0 - self.clip_epsilon)) | (b_ratios > (1.0 + self.clip_epsilon))
                    clip_fraction = torch.mean(clipped.float()).item()

                total_loss_accum += total_loss.item()
                policy_loss_accum += policy_loss.item()
                value_loss_accum += value_loss.item()
                entropy_accum += entropy.item()
                kl_accum += approx_kl
                clip_fraction_accum += clip_fraction
                update_count += 1

        return {
            "total_loss": total_loss_accum / update_count,
            "policy_loss": policy_loss_accum / update_count,
            "value_loss": value_loss_accum / update_count,
            "entropy": entropy_accum / update_count,
            "approximate_KL": kl_accum / update_count,
            "clip_fraction": clip_fraction_accum / update_count,
            "number_of_updates": update_count,
            "sequences_processed": B_total,
            "sequence_length": self.sequence_length
        }
