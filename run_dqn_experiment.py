import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless systems (still works fine on Windows but prevents display popup issues)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local 'pokemon_rl' package is discoverable from the script's root path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.dqn import select_action, DQNetwork
from pokemon_rl.agents.dqn_trainer import DQNTrainer

# Define hyperparameters
EPISODES = 1000
LEARNING_RATE = 1e-4
GAMMA = 0.99
BUFFER_CAPACITY = 10000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.05
TARGET_UPDATE_FREQ = 10
MAX_STEPS_PER_EPISODE = 200
EVAL_GAMES = 100

def print_hardware_info():
    """Detects and prints environment hardware info."""
    print("======================================================================")
    print("                DETECTING HARDWARE & PYTORCH SYSTEM                  ")
    print("======================================================================")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    if cuda_available:
        print(f"CUDA Version:    {torch.version.cuda}")
        print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    else:
         print("GPU Name:        N/A (CPU only)")
    print("======================================================================\n")

def evaluate_agent(env: PokemonTCGEnv, model: torch.nn.Module, device: torch.device, num_games: int, is_baseline: bool = False) -> Tuple[float, float, float]:
    """Runs evaluation games using either greedy action selection or a pure random baseline.
    
    Returns:
        win_rate (float), avg_reward (float), avg_length (float)
    """
    wins = 0
    rewards = []
    lengths = []
    
    # Put model in eval mode for inference
    model.eval()
    
    for game_idx in range(num_games):
        # Seed evaluation games to ensure reproducible benchmarks
        obs, info = env.reset(seed=9999 + game_idx)
        game_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < MAX_STEPS_PER_EPISODE:
            mask = env.get_action_mask()
            
            if is_baseline:
                # Epsilon = 1.0 forces select_action to strictly pick random legal moves
                action_idx = select_action(obs, model, mask, epsilon=1.0, device=device)
            else:
                # Epsilon = 0.0 forces strict greedy/exploitation selection among legal moves
                action_idx = select_action(obs, model, mask, epsilon=0.0, device=device)
                
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            game_reward += reward
            steps += 1
            
        winner = env.state.winner_idx
        if winner == 0:  # Player 0 (Red) is the Agent
            wins += 1
            
        rewards.append(game_reward)
        lengths.append(steps)
        
    win_rate = (wins / num_games) * 100.0
    return win_rate, float(np.mean(rewards)), float(np.mean(lengths))

def main():
    print_hardware_info()
    
    # Establish device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for experiment: {device}\n")
    
    # Ensure local directory is set up correctly
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Initialize Single-Agent Environment
    env = PokemonTCGEnv()
    
    # 2. Evaluate Untrained / Random Baseline
    print("Evaluating Untrained/Random Baseline (100 Games)...")
    dummy_model = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    baseline_win_rate, baseline_avg_reward, baseline_avg_len = evaluate_agent(
        env, dummy_model, device, EVAL_GAMES, is_baseline=True
    )
    print(f"Baseline Random Win Rate: {baseline_win_rate:.1f}%")
    print(f"Baseline Average Reward:  {baseline_avg_reward:.2f}")
    print(f"Baseline Average Length:  {baseline_avg_len:.1f} steps\n")
    
    # 3. Initialize DQNTrainer with Hyperparameters
    trainer = DQNTrainer(
        state_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        device=device
    )
    
    epsilon = EPSILON_START
    best_win_rate = 0.0
    rolling_window = 100
    
    history = {
        "episode": [],
        "cumulative_reward": [],
        "average_loss": [],
        "epsilon": [],
        "episode_length": [],
        "rolling_100_win_rate": [],
        "win": []
    }
    
    print(f"Training DQN on {device} for {EPISODES} episodes...")
    start_time = time.time()
    
    # 4. Main Training Loop
    for episode in range(1, EPISODES + 1):
        obs, info = env.reset(seed=42 + episode)
        
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        terminated = False
        truncated = False
        
        # Train policy net
        trainer.policy_net.train()
        
        while not (terminated or truncated) and episode_steps < MAX_STEPS_PER_EPISODE:
            mask = env.get_action_mask()
            
            # Select action epsilon-greedily (strictly matching legal action choices)
            action_idx = select_action(obs, trainer.policy_net, mask, epsilon, device)
            
            # Step Game Engine
            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            
            # Store experience transition
            trainer.replay_buffer.add(obs, action_idx, reward, next_obs, terminated)
            
            # Run one SGD optimization step
            loss_val = trainer.train_step()
            if loss_val is not None:
                episode_losses.append(loss_val)
                
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        winner = env.state.winner_idx
        agent_won = 1.0 if winner == 0 else 0.0
        
        # Update exploration decay
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Periodic target network synchronization
        if episode % TARGET_UPDATE_FREQ == 0:
            trainer.update_target_network()
            
        # Log metrics
        history["episode"].append(episode)
        history["cumulative_reward"].append(episode_reward)
        history["average_loss"].append(avg_loss)
        history["epsilon"].append(epsilon)
        history["episode_length"].append(episode_steps)
        history["win"].append(agent_won)
        
        # Compute rolling win rate
        recent_wins = history["win"][-rolling_window:]
        rolling_win_rate = np.mean(recent_wins) if recent_wins else 0.0
        history["rolling_100_win_rate"].append(rolling_win_rate)
        
        # Save checkpoints
        # Wait until we have a representative sample (at least 5 episodes) before saving the best
        if episode >= 5 and rolling_win_rate > best_win_rate:
            best_win_rate = rolling_win_rate
            torch.save(trainer.policy_net.state_dict(), os.path.join(results_dir, "best_model.pt"))
            
        if episode % 100 == 0:
            print(f"Episode: {episode:4d} | Reward: {episode_reward:5.1f} | Loss: {avg_loss:8.6f} | "
                  f"Epsilon: {epsilon:4.2f} | Win Rate (last {len(recent_wins)}): {rolling_win_rate:4.2f}")
                  
    training_time = time.time() - start_time
    print(f"\nTraining Complete! Time taken: {training_time:.1f} seconds.")
    
    # Save final model state dict
    torch.save(trainer.policy_net.state_dict(), os.path.join(results_dir, "final_model.pt"))
    
    # Save training metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_csv_path = os.path.join(results_dir, "dqn_training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics successfully saved to: {metrics_csv_path}")
    
    # 5. Evaluate Trained DQN (Using the best checkpointed weights)
    print("\nLoading Best DQN checkpoint for Greedy Evaluation...")
    best_model = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    best_model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pt"), map_location=device))
    
    print("Evaluating Trained DQN (100 Greedy Games)...")
    dqn_win_rate, dqn_avg_reward, dqn_avg_len = evaluate_agent(
        env, best_model, device, EVAL_GAMES, is_baseline=False
    )
    
    dqn_improvement = dqn_win_rate - baseline_win_rate
    
    # 6. Generate Performance Plots
    print("\nGenerating Performance Plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Performance Analysis - Mini Pokémon TCG RL", fontsize=16, fontweight='bold', y=0.96)
    
    # Top-Left: Training Reward
    axs[0, 0].plot(history["episode"], history["cumulative_reward"], color='lightblue', alpha=0.5, label='Episode Reward')
    # Rolling 50-episode average for visualization smoothing
    rolling_r = pd.Series(history["cumulative_reward"]).rolling(50, min_periods=1).mean()
    axs[0, 0].plot(history["episode"], rolling_r, color='royalblue', linewidth=2, label='50-Ep Rolling Mean')
    axs[0, 0].set_title("Training Reward vs. Episode", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Cumulative Reward")
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()
    
    # Top-Right: Rolling Win Rate
    axs[0, 1].plot(history["episode"], history["rolling_100_win_rate"], color='forestgreen', linewidth=2, label='100-Ep Rolling Win Rate')
    axs[0, 1].axhline(y=baseline_win_rate/100.0, color='red', linestyle='--', alpha=0.8, label=f'Random Baseline ({baseline_win_rate:.0f}%)')
    axs[0, 1].set_title("Rolling 100-Episode Win Rate vs. Episode", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Win Rate (Fraction)")
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    axs[0, 1].legend()
    
    # Bottom-Left: Loss Curve
    axs[1, 0].plot(history["episode"], history["average_loss"], color='darkorange', alpha=0.8)
    axs[1, 0].set_title("Training Huber Loss vs. Episode", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Huber Loss")
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Bottom-Right: Comparison Bar Chart
    categories = ['Random Baseline', 'Trained DQN (Best)']
    win_rates = [baseline_win_rate, dqn_win_rate]
    bars = axs[1, 1].bar(categories, win_rates, color=['salmon', 'dodgerblue'], width=0.5)
    axs[1, 1].set_title("Greedy Win Rate Evaluation Comparison", fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel("Win Rate (%)")
    axs[1, 1].set_ylim(0, 110)
    axs[1, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        axs[1, 1].annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
                            
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plot_path = os.path.join(results_dir, "dqn_performance_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance plots successfully saved to: {plot_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("                  TCG EXPERIMENT FINAL SUMMARY                        ")
    print("="*70)
    print(f"Device Used:                    {device}")
    print(f"Total Training Time:            {training_time:.1f} seconds")
    print(f"Baseline Random Win Rate:       {baseline_win_rate:.1f}%")
    print(f"DQN Win Rate:                   {dqn_win_rate:.1f}%")
    print(f"DQN Improvement:               +{dqn_improvement:.1f} percentage points")
    print(f"Baseline Avg. Game Length:      {baseline_avg_len:.1f} steps")
    print(f"DQN Avg. Game Length:           {dqn_avg_len:.1f} steps")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
