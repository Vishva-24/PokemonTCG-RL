import os
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.dqn import DQNetwork, select_action
from pokemon_rl.agents.dqn_trainer import DQNTrainer

class DQNConfig:
    """Configuration class for DQN training hyperparameters."""
    def __init__(self, **kwargs):
        self.num_episodes: int = kwargs.get("num_episodes", 1000)
        self.lr: float = kwargs.get("lr", 1e-4)
        self.gamma: float = kwargs.get("gamma", 0.99)
        self.buffer_capacity: int = kwargs.get("buffer_capacity", 10000)
        self.batch_size: int = kwargs.get("batch_size", 32)
        self.epsilon_start: float = kwargs.get("epsilon_start", 1.0)
        self.epsilon_decay: float = kwargs.get("epsilon_decay", 0.99)
        self.epsilon_min: float = kwargs.get("epsilon_min", 0.05)
        self.target_update_freq: int = kwargs.get("target_update_freq", 10)
        self.max_steps_per_episode: int = kwargs.get("max_steps_per_episode", 200)
        self.checkpoint_dir: str = kwargs.get("checkpoint_dir", "/workspace/scratch/checkpoints")

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def train_dqn(
    env: PokemonTCGEnv,
    config: DQNConfig,
    verbose: bool = True
) -> Tuple[DQNTrainer, Dict[str, List[Any]]]:
    """Runs the complete DQN training loop over a specified number of episodes.
    
    Returns:
        The trained DQNTrainer instance and a dictionary containing training history.
    """
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device("cpu")

    if env.observation_space.shape != (OBSERVATION_DIM,) or env.action_space.n != ACTION_DIM:
        raise ValueError(
            f"DQN requires the Phase 2 environment interface ({OBSERVATION_DIM} observations, {ACTION_DIM} actions)."
        )
    
    # Initialize the DQNTrainer with the configurations
    trainer = DQNTrainer(
        state_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        lr=config.lr,
        gamma=config.gamma,
        capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        device=device
    )

    epsilon = config.epsilon_start
    best_win_rate = 0.0
    rolling_window = 100
    
    # History tracking
    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "losses": [],
        "epsilons": [],
        "results": []  # 1.0 for Agent Win (Red), 0.0 for Agent Loss (Blue)
    }

    for episode in range(1, config.num_episodes + 1):
        # Reset environment with a clean seed sequence for tracking reproducibility
        obs, info = env.reset(seed=42 + episode)
        
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_steps < config.max_steps_per_episode:
            # 1. Get action mask for current state
            action_mask = env.get_action_mask()
            
            # 2. Select action epsilon-greedily
            action_idx = select_action(
                state=obs,
                model=trainer.policy_net,
                action_mask=action_mask,
                epsilon=epsilon,
                device=device
            )
            
            # 3. Step the environment
            next_obs, reward, terminated, truncated, step_info = env.step(action_idx)
            
            # 4. Save experience in replay buffer
            trainer.replay_buffer.add(
                state=obs,
                action=action_idx,
                reward=reward,
                next_state=next_obs,
                done=terminated
            )
            
            # 5. Optimize online network if buffer is sufficiently full
            loss_val = trainer.train_step()
            if loss_val is not None:
                episode_losses.append(loss_val)
                
            obs = next_obs
            episode_reward += reward
            episode_steps += 1

        # Calculate average loss for the episode
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        # Win-loss detection
        winner = env.state.winner_idx  # 0 is Red (Agent), 1 is Blue (Opponent)
        agent_won = 1.0 if winner == 0 else 0.0
        
        # Record statistics
        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(episode_steps)
        history["losses"].append(avg_loss)
        history["epsilons"].append(epsilon)
        history["results"].append(agent_won)
        
        # Decaying epsilon
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        
        # Update target network periodically
        if episode % config.target_update_freq == 0:
            trainer.update_target_network()
            
        # Calculate rolling win rate
        recent_results = history["results"][-rolling_window:]
        rolling_win_rate = np.mean(recent_results) if recent_results else 0.0
        
        # Save best model checkpoint if win rate improves
        # Ensure we have at least a few episodes (e.g. 5) before evaluating best win rate,
        # and it's strictly better than the historical best.
        if episode >= 5 and rolling_win_rate > best_win_rate:
            best_win_rate = rolling_win_rate
            checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save(trainer.policy_net.state_dict(), checkpoint_path)
            if verbose and episode > 10: # Only log rolling best updates for substantial runs
                print(f"--> Episode {episode}: New best rolling win rate: {best_win_rate:.2f}. Checkpoint saved.")

        if verbose:
            print(f"Episode: {episode:3d} | Reward: {episode_reward:5.1f} | Loss: {avg_loss:8.6f} | "
                  f"Epsilon: {epsilon:4.2f} | Win Rate (last {len(recent_results)}): {rolling_win_rate:4.2f} | Winner: {env.state.players[winner].name if winner is not None else 'None'}")

    # Save final model checkpoint at the end of training
    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    torch.save(trainer.policy_net.state_dict(), final_checkpoint_path)
    
    return trainer, history
