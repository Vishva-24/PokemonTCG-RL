import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.lstm_ppo import LSTMActorNetwork, LSTMCriticNetwork, LSTMPPORolloutBuffer, MaskedCategorical
from pokemon_rl.agents.lstm_ppo_trainer import LSTMPPOTrainer, compute_recurrent_gae_and_returns

class LSTMPPOConfig:
    """Configuration class for LSTM-PPO training hyperparameters."""
    def __init__(self, **kwargs):
        self.num_updates: int = kwargs.get("num_updates", 500)
        self.rollout_steps: int = kwargs.get("rollout_steps", 2048)
        self.sequence_length: int = kwargs.get("sequence_length", 16)
        self.batch_size: int = kwargs.get("batch_size", 4)
        self.ppo_epochs: int = kwargs.get("ppo_epochs", 10)
        self.lr: float = kwargs.get("lr", 5e-5)
        self.gamma: float = kwargs.get("gamma", 0.99)
        self.gae_lambda: float = kwargs.get("gae_lambda", 0.95)
        self.clip_epsilon: float = kwargs.get("clip_epsilon", 0.2)
        self.entropy_coef: float = kwargs.get("entropy_coefficient", 0.01)
        self.vf_coef: float = kwargs.get("value_loss_coefficient", 0.5)
        self.max_grad_norm: float = kwargs.get("max_grad_norm", 0.5)
        self.checkpoint_dir: str = kwargs.get("checkpoint_dir", "results")

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def parse_recurrent_completed_episodes(buffer: LSTMPPORolloutBuffer) -> Tuple[List[float], List[int], List[float]]:
    """Identifies completed episode boundaries and calculates stats from the recurrent rollout buffer."""
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
            win_val = 1.0 if r == 1.0 else 0.0
            ep_wins.append(win_val)
            
            current_reward = 0.0
            current_len = 0
            
    return ep_rewards, ep_lengths, ep_wins


def train_lstm_ppo(
    env: PokemonTCGEnv,
    config: LSTMPPOConfig,
    verbose: bool = True
) -> Tuple[LSTMActorNetwork, LSTMCriticNetwork, Dict[str, List[Any]]]:
    """Runs the complete LSTM-PPO training loop with recurrent hidden state management."""
    if env.observation_space.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"LSTM-PPO requires Phase 2 observations shaped ({OBSERVATION_DIM},), "
            f"got {env.observation_space.shape}."
        )
    if env.action_space.n != ACTION_DIM:
        raise ValueError(
            f"LSTM-PPO requires the Phase 2 {ACTION_DIM}-action space, got {env.action_space.n}."
        )

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("======================================================================")
        print("                LSTM-PPO TRAINING HARDWARE DETECTION                  ")
        print("======================================================================")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
        print(f"Selected Device: {device}")
        print("======================================================================\n")

    # 2. Instantiate networks
    actor = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    critic = LSTMCriticNetwork(state_dim=OBSERVATION_DIM).to(device)

    # 3. Instantiate Rollout Buffer and Trainer
    buffer = LSTMPPORolloutBuffer()
    trainer = LSTMPPOTrainer(
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
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        device=device
    )

    history = {
        "update": [],
        "total_steps": [],
        "num_episodes": [],
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
    
    # Initialize recurrent states
    h_actor, c_actor = actor.init_hidden(batch_size=1, device=device)
    h_critic, c_critic = critic.init_hidden(batch_size=1, device=device)
    
    current_obs, current_info = env.reset(seed=2000)
    current_obs = np.asarray(current_obs, dtype=np.float32)
    seed_sequence = 2001

    for update_idx in range(1, config.num_updates + 1):
        completed_ep_count = 0
        
        # Collect exactly config.rollout_steps transitions
        for step_idx in range(config.rollout_steps):
            if current_obs.shape != (OBSERVATION_DIM,):
                raise ValueError(
                    f"LSTM-PPO rollout received observation shape {current_obs.shape}; "
                    f"expected ({OBSERVATION_DIM},)."
                )
            state_t = torch.tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            mask = np.asarray(env.get_action_mask(), dtype=np.float32)
            if mask.shape != (ACTION_DIM,):
                raise ValueError(
                    f"LSTM-PPO rollout received action mask shape {mask.shape}; "
                    f"expected ({ACTION_DIM},)."
                )
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            # Keep copies of the hidden states BEFORE this transition
            h_actor_step = h_actor.clone()
            c_actor_step = c_actor.clone()
            h_critic_step = h_critic.clone()
            c_critic_step = c_critic.clone()
            
            with torch.no_grad():
                logits, (next_h_actor, next_c_actor) = actor(state_t, (h_actor, c_actor))
                value_tensor, (next_h_critic, next_c_critic) = critic(state_t, (h_critic, c_critic))
                value = value_tensor.squeeze().item()
                
                policy = MaskedCategorical(logits, mask_t)
                action_t = policy.sample()
                action = action_t.squeeze().item()
                log_prob = policy.log_prob(action_t).squeeze().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store in rollout buffer with both Actor and Critic states
            buffer.add(
                state=current_obs,
                action=action,
                reward=reward,
                done=terminated,
                value=value,
                log_prob=log_prob,
                action_mask=mask,
                actor_h=h_actor_step,
                actor_c=c_actor_step,
                critic_h=h_critic_step,
                critic_c=c_critic_step
            )

            if done:
                current_obs, current_info = env.reset(seed=seed_sequence)
                current_obs = np.asarray(current_obs, dtype=np.float32)
                seed_sequence += 1
                completed_ep_count += 1
                
                # Hard reset recurrent hidden states to zeros
                h_actor, c_actor = actor.init_hidden(batch_size=1, device=device)
                h_critic, c_critic = critic.init_hidden(batch_size=1, device=device)
            else:
                current_obs = np.asarray(next_obs, dtype=np.float32)
                current_info = info
                h_actor, c_actor = next_h_actor, next_c_actor
                h_critic, c_critic = next_h_critic, next_c_critic

        total_env_steps += config.rollout_steps

        ep_rewards, ep_lengths, ep_wins = parse_recurrent_completed_episodes(buffer)
        all_wins_history.extend(ep_wins)

        if len(buffer) > 0 and buffer.dones[-1]:
            last_value = 0.0
        else:
            last_obs_t = torch.tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                val_tensor, _ = critic(last_obs_t, (h_critic, c_critic))
                last_value = val_tensor.squeeze().item()

        # Extract sequence batches
        states_seq, action_masks_seq, actions_seq, rewards_seq, dones_seq, values_seq, log_probs_seq, act_h_init, act_c_init, cr_h_init, cr_c_init = \
            buffer.get_sequence_batches(sequence_length=config.sequence_length, device=device)
            
        rewards_flat = rewards_seq.cpu().numpy().flatten()
        dones_flat = dones_seq.cpu().numpy().flatten()
        values_flat = values_seq.cpu().numpy().flatten()
        
        advantages_flat, returns_flat = compute_recurrent_gae_and_returns(
            rewards=rewards_flat,
            dones=dones_flat,
            values=values_flat,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize_advantages=True
        )
        
        num_seqs = states_seq.shape[0]
        advantages_seq = torch.tensor(advantages_flat.reshape(num_seqs, config.sequence_length), dtype=torch.float32, device=device)
        returns_seq = torch.tensor(returns_flat.reshape(num_seqs, config.sequence_length), dtype=torch.float32, device=device)

        # Train recurrent networks with both Actor and Critic initial recurrent states and dones sequence
        diagnostics = trainer.update(
            states=states_seq,
            action_masks=action_masks_seq,
            actions=actions_seq,
            old_log_probs=log_probs_seq,
            returns=returns_seq,
            advantages=advantages_seq,
            actor_h0=act_h_init,
            actor_c0=act_c_init,
            critic_h0=cr_h_init,
            critic_c0=cr_c_init,
            dones=dones_seq
        )

        buffer.clear()

        avg_ep_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        avg_ep_len = np.mean(ep_lengths) if ep_lengths else 0.0
        
        recent_wins = all_wins_history[-rolling_window:]
        rolling_win_rate = np.mean(recent_wins) if recent_wins else 0.0

        history["update"].append(update_idx)
        history["total_steps"].append(total_env_steps)
        history["num_episodes"].append(completed_ep_count)
        history["avg_episode_reward"].append(avg_ep_reward)
        history["avg_episode_length"].append(avg_ep_len)
        history["win_rate"].append(rolling_win_rate)
        history["policy_loss"].append(diagnostics["policy_loss"])
        history["value_loss"].append(diagnostics["value_loss"])
        history["entropy"].append(diagnostics["entropy"])
        history["approximate_KL"].append(diagnostics["approximate_KL"])
        history["clip_fraction"].append(diagnostics["clip_fraction"])

        if len(all_wins_history) >= 5 and rolling_win_rate > best_win_rate:
            best_win_rate = rolling_win_rate
            torch.save(actor.state_dict(), os.path.join(config.checkpoint_dir, "best_lstm_ppo_actor.pt"))
            torch.save(critic.state_dict(), os.path.join(config.checkpoint_dir, "best_lstm_ppo_critic.pt"))
            if verbose and update_idx > 5:
                print(f"--> Update {update_idx:3d}: New best rolling win rate: {best_win_rate * 100.0:.1f}%. Checkpoint saved.")

        if verbose and (update_idx == 1 or update_idx % 10 == 0 or update_idx == config.num_updates):
            print(f"Update: {update_idx:3d} | Steps: {total_env_steps:6d} | "
                  f"Avg Reward: {avg_ep_reward:5.2f} | Avg Length: {avg_ep_len:4.1f} | "
                  f"Win Rate: {rolling_win_rate * 100.0:5.1f}% | Policy Loss: {diagnostics['policy_loss']:8.5f} | "
                  f"Value Loss: {diagnostics['value_loss']:8.5f} | Entropy: {diagnostics['entropy']:5.3f}")

    torch.save(actor.state_dict(), os.path.join(config.checkpoint_dir, "final_lstm_ppo_actor.pt"))
    torch.save(critic.state_dict(), os.path.join(config.checkpoint_dir, "final_lstm_ppo_critic.pt"))

    return actor, critic, history
