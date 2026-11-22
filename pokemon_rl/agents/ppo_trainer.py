import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from pokemon_rl.agents.ppo import ActorNetwork, CriticNetwork, MaskedCategorical
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

class PPOTrainer:
    """Trainer orchestrator for PPO, managing networks, separate optimizers, and SGD updates."""
    def __init__(
        self,
        actor: ActorNetwork,
        critic: CriticNetwork,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coefficient: float = 0.01,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
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
        self.batch_size = batch_size
        self.device = device

        # Separate optimizers for Actor and Critic to preserve learning stability
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.loss_fn = nn.MSELoss()

    def _validate_update_inputs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> None:
        """Validate the fixed Phase 2 PPO rollout interface before optimization."""
        if states.ndim != 2 or states.shape[1] != OBSERVATION_DIM:
            raise ValueError(
                f"PPO states must have shape (T, {OBSERVATION_DIM}), got {tuple(states.shape)}."
            )

        dataset_size = states.shape[0]
        if dataset_size == 0:
            raise ValueError("PPO cannot update from an empty rollout.")
        if actions.shape != (dataset_size,):
            raise ValueError(f"PPO actions must have shape ({dataset_size},), got {tuple(actions.shape)}.")
        if actions.dtype != torch.long:
            raise ValueError("PPO actions must be torch.long action indices.")
        for name, tensor in {
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
        }.items():
            if tensor.shape != (dataset_size,):
                raise ValueError(
                    f"PPO {name} must have shape ({dataset_size},), got {tuple(tensor.shape)}."
                )
        if action_masks.shape != (dataset_size, ACTION_DIM):
            raise ValueError(
                f"PPO action masks must have shape ({dataset_size}, {ACTION_DIM}), "
                f"got {tuple(action_masks.shape)}."
            )
        if not torch.all((action_masks == 0) | (action_masks == 1)):
            raise ValueError("PPO action masks must be binary.")
        if torch.any(action_masks.sum(dim=1) == 0):
            raise ValueError("Each PPO action mask must contain at least one legal action.")
        if torch.any(actions < 0) or torch.any(actions >= ACTION_DIM):
            raise ValueError(f"PPO actions must be in [0, {ACTION_DIM - 1}].")

        batch_indices = torch.arange(dataset_size, device=actions.device)
        if torch.any(action_masks[batch_indices, actions] != 1):
            raise ValueError("PPO rollout contains an action that is illegal under its stored mask.")

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        action_masks: torch.Tensor
    ) -> Dict[str, Any]:
        """Performs PPO updates over multiple epochs and minibatches.
        
        Args:
            states: PyTorch Tensor of shape (T, 50)
            actions: PyTorch Tensor of shape (T,)
            old_log_probs: PyTorch Tensor of shape (T,)
            returns: PyTorch Tensor of shape (T,)
            advantages: PyTorch Tensor of shape (T,)
            action_masks: PyTorch Tensor of shape (T, 70)
            
        Returns:
            A dictionary containing diagnostics of the updates.
        """
        # Ensure all tensors reside on the active device and have zero gradients
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device).detach()
        returns = returns.to(self.device).detach()
        advantages = advantages.to(self.device).detach()
        action_masks = action_masks.to(self.device)

        self._validate_update_inputs(
            states, actions, old_log_probs, returns, advantages, action_masks
        )

        dataset_size = states.shape[0]
        
        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        entropy_accum = 0.0
        kl_accum = 0.0
        clip_fraction_accum = 0.0
        update_count = 0

        # Perform ppo_epochs full passes over the dataset
        for epoch in range(self.ppo_epochs):
            # Shuffle data indices for stochastic minibatch gradient descent
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]
                b_action_masks = action_masks[batch_indices]

                # 1. Run Actor on states to obtain new logits
                b_logits = self.actor(b_states)

                # 2. Apply action masks and obtain Masked Categorical policy distribution
                b_policy = MaskedCategorical(b_logits, b_action_masks)

                # 3. Compute new log probabilities for action index taken
                b_new_log_probs = b_policy.log_prob(b_actions)

                # 4. Compute probability ratio r_t = exp(new_log_prob - old_log_prob)
                b_ratios = torch.exp(b_new_log_probs - b_old_log_probs)

                # 5. Compute clipped surrogate policy objective
                surr1 = b_ratios * b_advantages
                surr2 = torch.clamp(b_ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                b_l_clip = torch.min(surr1, surr2)

                # 6. Policy Loss = -mean(L_clip)
                policy_loss = -torch.mean(b_l_clip)

                # 7. Run Critic on states to obtain value predictions and compute value loss
                b_new_values = self.critic(b_states).squeeze(-1)
                value_loss = self.loss_fn(b_new_values, b_returns)

                # 8. Compute policy entropy
                entropy = torch.mean(b_policy.entropy())

                # 9. Combined optimization objective (conceptually)
                total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                # 10. Update Actor parameters
                self.actor_optimizer.zero_grad()
                actor_loss = policy_loss - self.entropy_coef * entropy
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 11. Update Critic parameters
                self.critic_optimizer.zero_grad()
                critic_loss = self.vf_coef * value_loss
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 12. Compute diagnostics for tracking
                with torch.no_grad():
                    # Approximate KL (stable SB3 estimator)
                    log_ratio = b_new_log_probs - b_old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).item()

                    # Clip Fraction
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
            "number_of_updates": update_count
        }
