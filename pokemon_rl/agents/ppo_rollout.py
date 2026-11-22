import torch
import numpy as np
from typing import Tuple, Optional

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.ppo import MaskedCategorical, PPORolloutBuffer

def collect_rollout(
    env: PokemonTCGEnv,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    buffer: PPORolloutBuffer,
    rollout_steps: int,
    device: torch.device,
    current_obs: Optional[np.ndarray] = None,
    current_info: Optional[dict] = None,
    seed_sequence: int = 1000
) -> Tuple[np.ndarray, dict, int, float]:
    """Collects exactly rollout_steps transitions from the environment using the actor/critic policies.
    
    PPO controls Red (Player 0) as a single agent, and the opponent plays randomly.
    If an episode ends (terminated or truncated), the environment resets and collection continues.
    
    Args:
        env: The PokemonTCGEnv environment instance.
        actor: Actor neural network.
        critic: Critic neural network.
        buffer: PPORolloutBuffer to store transitions.
        rollout_steps: Number of transitions to collect.
        device: CPU or CUDA GPU device.
        current_obs: The starting observation (optional).
        current_info: The starting info (optional).
        seed_sequence: The seed counter for environment resets.
        
    Returns:
        next_obs (np.ndarray): The next state observation for continuing collection.
        next_info (dict): The next info dictionary.
        seed_sequence (int): The updated seed counter.
        last_value (float): The critic value estimate for the next observation state.
    """
    actor.eval()
    critic.eval()

    if env.observation_space.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"PPO requires Phase 2 observations shaped ({OBSERVATION_DIM},), "
            f"got {env.observation_space.shape}."
        )
    if env.action_space.n != ACTION_DIM:
        raise ValueError(
            f"PPO requires the Phase 2 {ACTION_DIM}-action space, got {env.action_space.n}."
        )
    
    if current_obs is None:
        current_obs, current_info = env.reset(seed=seed_sequence)
        seed_sequence += 1

    current_obs = np.asarray(current_obs, dtype=np.float32)
    if current_obs.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"PPO rollout received an observation shaped {current_obs.shape}; "
            f"expected ({OBSERVATION_DIM},)."
        )
        
    for step_idx in range(rollout_steps):
        # Convert state observation to tensor
        state_t = torch.tensor(current_obs, dtype=torch.float32, device=device)
        
        # Get legal action mask
        mask = np.asarray(env.get_action_mask(), dtype=np.float32)
        if mask.shape != (ACTION_DIM,):
            raise ValueError(
                f"PPO rollout received an action mask shaped {mask.shape}; "
                f"expected ({ACTION_DIM},)."
            )
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Get logits and critic state value V(s)
            # Unsqueeze if networks require batch dim, but our networks support single vector input
            logits = actor(state_t)
            value = critic(state_t).item()
            
            # Mask logits and sample legal action
            policy = MaskedCategorical(logits, mask_t)
            action_t = policy.sample()
            action = action_t.item()
            log_prob = policy.log_prob(action_t).item()
            
        # Step the environment with sampled action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Save transitions strictly to our rollout buffer (using standard terminated for boot-zero)
        buffer.add(
            state=current_obs,
            action=action,
            reward=reward,
            done=terminated,
            value=value,
            log_prob=log_prob,
            action_mask=mask
        )
        
        if done:
            # Reset on boundary, increment seed sequence
            current_obs, current_info = env.reset(seed=seed_sequence)
            seed_sequence += 1
        else:
            current_obs = next_obs
            current_info = info
            
    # Estimate value of the final next state for GAE bootstrapping
    # If the last collected step was terminal, bootstrap is zeroed by the done signal, so value can be 0.0
    if len(buffer) > 0 and buffer.dones[-1]:
        last_value = 0.0
    else:
        last_obs_t = torch.tensor(current_obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            last_value = critic(last_obs_t).item()
            
    return current_obs, current_info, seed_sequence, last_value


def compute_gae_and_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Generalized Advantage Estimation (GAE) and Discounted Target Returns.
    
    Args:
        rewards: numpy array of shape (T,)
        dones: numpy array of shape (T,) containing 1.0 for terminated and 0.0 otherwise
        values: numpy array of shape (T,) containing critic state value V(s)
        last_value: critic state value V(s_{T+1}) of the state immediately following the last transition
        gamma: discount factor
        gae_lambda: GAE decay factor lambda
        normalize_advantages: standardizes advantages to mean 0, std 1
        
    Returns:
        advantages (np.ndarray of shape (T,)): GAE advantages
        returns (np.ndarray of shape (T,)): Target returns for optimizing the Critic
    """
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
            
        # TD Residual delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_non_terminal * next_value - values[t]
        
        # Generalized Advantage A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
    # Return = Advantage + Value
    returns = advantages + values
    
    if normalize_advantages and T > 1:
        std = advantages.std()
        mean = advantages.mean()
        advantages = (advantages - mean) / (std + 1e-8)
        
    return advantages, returns
