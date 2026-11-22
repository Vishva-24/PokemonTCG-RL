import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.ppo import ActorNetwork, CriticNetwork, PPORolloutBuffer
from pokemon_rl.agents.ppo_rollout import collect_rollout, compute_gae_and_returns
from pokemon_rl.agents.ppo_trainer import PPOTrainer

class PPOConfig:
    """Configuration class for PPO training hyperparameters."""
    def __init__(self, **kwargs):
        self.num_updates: int = kwargs.get("num_updates", 500)
        self.rollout_steps: int = kwargs.get("rollout_steps", 2048)
        self.ppo_epochs: int = kwargs.get("ppo_epochs", 10)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.lr: float = kwargs.get("lr", 1e-4)
        self.gamma: float = kwargs.get("gamma", 0.99)
        self.gae_lambda: float = kwargs.get("gae_lambda", 0.95)
        self.clip_epsilon: float = kwargs.get("clip_epsilon", 0.2)
        self.entropy_coef: float = kwargs.get("entropy_coefficient", 0.01)
        self.vf_coef: float = kwargs.get("value_loss_coefficient", 0.5)
        self.max_grad_norm: float = kwargs.get("max_grad_norm", 0.5)
        self.checkpoint_dir: str = kwargs.get("checkpoint_dir", "results")

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def parse_completed_episodes(buffer: PPORolloutBuffer) -> Tuple[List[float], List[int], List[float]]:
    """Helper to reconstruct completed episode statistics from a rollout buffer.
    
    Identifies boundaries where done is True.
    - episode reward is the sum of step rewards.
    - episode length is the count of steps.
    - win is 1.0 if the final terminal reward is 1.0 (Red won), and 0.0 if -1.0 (Blue won).
    """
    rewards = buffer.rewards
    dones = buffer.dones
    
    ep_rewards = []
    ep_lengths = []
    ep_wins = []
    
    current_reward = 0.0
    current_len = 0
    
    for r, d in zip(rewards, dones):
        current_reward += r
        current_len += 1
        if d:
            ep_rewards.append(current_reward)
            ep_lengths.append(current_len)
            # Terminal reward of 1.0 means Player 0 (Red/PPO) won.
            # -1.0 means Player 1 (Blue) won.
            win_val = 1.0 if r == 1.0 else 0.0
            ep_wins.append(win_val)
            
            # Reset trackers
            current_reward = 0.0
            current_len = 0
            
    return ep_rewards, ep_lengths, ep_wins


def train_ppo(
    env: PokemonTCGEnv,
    config: PPOConfig,
    verbose: bool = True
) -> Tuple[ActorNetwork, CriticNetwork, Dict[str, List[Any]]]:
    """Runs the complete PPO training loop.
    
    PPO plays as Player 0 (Red) against Player 1 (Blue), who plays randomly.
    """
    if env.observation_space.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"PPO requires Phase 2 observations shaped ({OBSERVATION_DIM},), "
            f"got {env.observation_space.shape}."
        )
    if env.action_space.n != ACTION_DIM:
        raise ValueError(
            f"PPO requires the Phase 2 {ACTION_DIM}-action space, got {env.action_space.n}."
        )

    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 1. Automatic hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("======================================================================")
        print("                PPO TRAINING HARDWARE DETECTION                      ")
        print("======================================================================")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
        print(f"Selected Device: {device}")
        print("======================================================================\n")

    # 2. Instantiate networks
    actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    critic = CriticNetwork(state_dim=OBSERVATION_DIM).to(device)

    # 3. Instantiate PPORolloutBuffer and PPOTrainer
    buffer = PPORolloutBuffer()
    trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        lr=config.lr,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        entropy_coefficient=config.entropy_coef,
        value_loss_coefficient=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        ppo_epochs=config.ppo_epochs,
        batch_size=config.batch_size,
        device=device
    )

    # 4. History and progress tracking
    history = {
        "update": [],
        "total_steps": [],
        "avg_episode_reward": [],
        "avg_episode_length": [],
        "win_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approximate_KL": [],
        "clip_fraction": []
    }

    total_env_steps = 0
    best_win_rate = 0.0
    rolling_window = 100
    all_wins_history = []
    
    # State tracking across rollout collections
    current_obs, current_info = env.reset(seed=1000)
    seed_sequence = 1001

    # 5. Core PPO Training Loop
    for update_idx in range(1, config.num_updates + 1):
        # a. Collect fresh on-policy rollout
        current_obs, current_info, seed_sequence, last_value = collect_rollout(
            env=env,
            actor=actor,
            critic=critic,
            buffer=buffer,
            rollout_steps=config.rollout_steps,
            device=device,
            current_obs=current_obs,
            current_info=current_info,
            seed_sequence=seed_sequence
        )
        
        total_env_steps += config.rollout_steps
        
        # Parse completed episodes in this rollout for win tracking and diagnostics
        ep_rewards, ep_lengths, ep_wins = parse_completed_episodes(buffer)
        all_wins_history.extend(ep_wins)
        
        # b. Compute GAE and returns
        states, actions, rewards, dones, values, log_probs, action_masks = buffer.get_tensors(device)
        
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()
        values_np = values.cpu().numpy()
        
        advantages_np, returns_np = compute_gae_and_returns(
            rewards=rewards_np,
            dones=dones_np,
            values=values_np,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize_advantages=True
        )
        
        advantages = torch.tensor(advantages_np, dtype=torch.float32, device=device)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=device)
        
        # c. Run PPO Trainer updates
        diagnostics = trainer.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            returns=returns,
            advantages=advantages,
            action_masks=action_masks
        )
        
        # d. Clear the rollout buffer
        buffer.clear()
        
        # Calculate evaluation metrics
        avg_ep_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        avg_ep_len = np.mean(ep_lengths) if ep_lengths else 0.0
        
        # Rolling win rate over the last 100 completed episodes
        recent_wins = all_wins_history[-rolling_window:]
        rolling_win_rate = np.mean(recent_wins) if recent_wins else 0.0
        
        # Record metrics in history
        history["update"].append(update_idx)
        history["total_steps"].append(total_env_steps)
        history["avg_episode_reward"].append(avg_ep_reward)
        history["avg_episode_length"].append(avg_ep_len)
        history["win_rate"].append(rolling_win_rate)
        history["policy_loss"].append(diagnostics["policy_loss"])
        history["value_loss"].append(diagnostics["value_loss"])
        history["entropy"].append(diagnostics["entropy"])
        history["approximate_KL"].append(diagnostics["approximate_KL"])
        history["clip_fraction"].append(diagnostics["clip_fraction"])
        
        # e. Save model checkpoints when rolling win rate improves
        # Ensure we have at least 5 completed episodes to calculate a meaningful win rate
        if len(all_wins_history) >= 5 and rolling_win_rate > best_win_rate:
            best_win_rate = rolling_win_rate
            torch.save(actor.state_dict(), os.path.join(config.checkpoint_dir, "best_model.pt"))
            if verbose and update_idx > 5:
                print(f"--> Update {update_idx:3d}: New best rolling win rate: {best_win_rate * 100.0:.1f}%. Checkpoint saved.")

        if verbose and (update_idx == 1 or update_idx % 10 == 0 or update_idx == config.num_updates):
            print(f"Update: {update_idx:3d} | Steps: {total_env_steps:6d} | "
                  f"Avg Reward: {avg_ep_reward:5.2f} | Avg Length: {avg_ep_len:4.1f} | "
                  f"Win Rate: {rolling_win_rate * 100.0:5.1f}% | Policy Loss: {diagnostics['policy_loss']:8.5f} | "
                  f"Value Loss: {diagnostics['value_loss']:8.5f} | Entropy: {diagnostics['entropy']:5.3f}")

    # Save final model state dict at end of training
    torch.save(actor.state_dict(), os.path.join(config.checkpoint_dir, "final_model.pt"))
    
    return actor, critic, history
