import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless / Windows execution safety
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

# Ensure local 'pokemon_rl' package is discoverable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.ppo import ActorNetwork, CriticNetwork, MaskedCategorical, PPORolloutBuffer
from pokemon_rl.agents.ppo_rollout import collect_rollout, compute_gae_and_returns
from pokemon_rl.agents.ppo_trainer import PPOTrainer

# Experiment Configurations
NUM_UPDATES = 500
ROLLOUT_STEPS = 2048
PPO_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
MAX_STEPS_PER_EPISODE = 200

SEED_RANGES = [
    (10000, 10499),
    (20000, 20499),
    (30000, 30499)
]

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

def evaluate_policy(
    env: PokemonTCGEnv, 
    actor: torch.nn.Module, 
    device: torch.device, 
    is_random_baseline: bool = False
) -> dict:
    """Evaluates a policy across the three configured seed ranges of 500 games each.
    
    Args:
        env: The PokemonTCGEnv environment instance.
        actor: Actor neural network.
        device: CPU or CUDA GPU device.
        is_random_baseline: If True, selects random legal actions.
        
    Returns:
        A dictionary containing breakdown metrics per range and overall metrics.
    """
    actor.eval()
    results = {
        "ranges": {},
        "overall": {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "total_steps": 0,
            "avg_length": 0.0
        }
    }
    
    for r_start, r_end in SEED_RANGES:
        range_label = f"{r_start}-{r_end}"
        num_games = r_end - r_start + 1
        
        wins = 0
        losses = 0
        total_reward = 0.0
        total_steps = 0
        
        for seed in range(r_start, r_end + 1):
            obs, info = env.reset(seed=seed)
            game_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < MAX_STEPS_PER_EPISODE:
                mask = env.get_action_mask()
                
                if is_random_baseline:
                    # Choose a random legal action
                    legal_indices = np.where(mask == 1.0)[0]
                    action_idx = int(np.random.choice(legal_indices))
                else:
                    # Greedy evaluation: select highest-probability legal action
                    state_t = torch.tensor(obs, dtype=torch.float32, device=device)
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits = actor(state_t)
                        policy = MaskedCategorical(logits, mask_t)
                        probs = policy.probs
                        action_idx = int(torch.argmax(probs).item())
                        
                obs, reward, terminated, truncated, _ = env.step(action_idx)
                game_reward += reward
                steps += 1
                
            winner = env.state.winner_idx
            if winner == 0:  # Agent (Red) won
                wins += 1
            else:
                losses += 1
                
            total_reward += game_reward
            total_steps += steps
            
        win_rate = (wins / num_games) * 100.0
        avg_reward = total_reward / num_games
        avg_length = total_steps / num_games
        
        results["ranges"][range_label] = {
            "games": num_games,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_length": avg_length
        }
        
        # Accumulate to overall metrics
        results["overall"]["games"] += num_games
        results["overall"]["wins"] += wins
        results["overall"]["losses"] += losses
        results["overall"]["total_reward"] += total_reward
        results["overall"]["total_steps"] += total_steps
        
    # Calculate overall averages
    o = results["overall"]
    o["win_rate"] = (o["wins"] / o["games"]) * 100.0
    o["avg_reward"] = o["total_reward"] / o["games"]
    o["avg_length"] = o["total_steps"] / o["games"]
    
    return results

def parse_completed_episodes(buffer: PPORolloutBuffer) -> Tuple[List[float], List[int], List[float]]:
    """Helper to reconstruct completed episode statistics from a rollout buffer."""
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

def main(is_test_run: bool = False):
    print_hardware_info()
    
    # Establish device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for experiment: {device}\n")
    
    # Ensure local directory is set up correctly beside the script
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Adjust configurations if this is a validation test run
    num_updates = 5 if is_test_run else NUM_UPDATES
    rollout_steps = 64 if is_test_run else ROLLOUT_STEPS
    batch_size = 16 if is_test_run else BATCH_SIZE
    
    # Temporary seed ranges override for testing
    global SEED_RANGES
    if is_test_run:
        SEED_RANGES = [(10000, 10004), (20000, 20004), (30000, 30004)]
        
    env = PokemonTCGEnv()
    
    # 1. Evaluate Untrained / Random Baseline
    print("Evaluating Untrained/Random Baseline (1500 Games)...")
    dummy_actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    baseline_results = evaluate_policy(env, dummy_actor, device, is_random_baseline=True)
    baseline_win_rate = baseline_results["overall"]["win_rate"]
    baseline_avg_reward = baseline_results["overall"]["avg_reward"]
    baseline_avg_len = baseline_results["overall"]["avg_length"]
    
    print(f"Baseline Random Win Rate: {baseline_win_rate:.1f}%")
    print(f"Baseline Average Reward:  {baseline_avg_reward:.2f}")
    print(f"Baseline Average Length:  {baseline_avg_len:.1f} steps\n")
    
    # 2. Initialize PPO Networks, Trainer, and Buffer
    actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    critic = CriticNetwork(state_dim=OBSERVATION_DIM).to(device)
    buffer = PPORolloutBuffer()
    
    trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        entropy_coefficient=ENTROPY_COEF,
        value_loss_coefficient=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        ppo_epochs=PPO_EPOCHS,
        batch_size=batch_size,
        device=device
    )
    
    # Metrics tracking
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
    
    print(f"Training PPO on {device} for {num_updates} updates...")
    start_time = time.time()
    
    # 3. Main PPO Training Loop
    for update_idx in range(1, num_updates + 1):
        # a. Collect fresh on-policy rollout
        current_obs, current_info, seed_sequence, last_value = collect_rollout(
            env=env,
            actor=actor,
            critic=critic,
            buffer=buffer,
            rollout_steps=rollout_steps,
            device=device,
            current_obs=current_obs,
            current_info=current_info,
            seed_sequence=seed_sequence
        )
        
        total_env_steps += rollout_steps
        
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
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
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
        if len(all_wins_history) >= 5 and rolling_win_rate > best_win_rate:
            best_win_rate = rolling_win_rate
            torch.save(actor.state_dict(), os.path.join(results_dir, "best_ppo_model.pt"))
            
        if update_idx == 1 or update_idx % 10 == 0 or update_idx == num_updates:
            print(f"Update: {update_idx:3d} | Steps: {total_env_steps:6d} | "
                  f"Avg Reward: {avg_ep_reward:5.2f} | Avg Length: {avg_ep_len:4.1f} | "
                  f"Win Rate: {rolling_win_rate * 100.0:5.1f}% | Policy Loss: {diagnostics['policy_loss']:8.5f} | "
                  f"Value Loss: {diagnostics['value_loss']:8.5f} | Entropy: {diagnostics['entropy']:5.3f}")
                  
    training_time = time.time() - start_time
    print(f"\nTraining Complete! Time taken: {training_time:.1f} seconds.")
    
    # Save final model state dict
    torch.save(actor.state_dict(), os.path.join(results_dir, "final_ppo_model.pt"))
    
    # Save training metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_csv_path = os.path.join(results_dir, "ppo_training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics successfully saved to: {metrics_csv_path}")
    
    # 4. Evaluate Trained PPO (Best and Final check-pointed models)
    print("\nLoading Best PPO checkpoint for Greedy Evaluation...")
    best_actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    final_model_path = os.path.join(results_dir, "final_ppo_model.pt")
    # Fall back to the newly trained final model if a best checkpoint was never
    # written, or if a stale pre-Phase-2 18-action checkpoint is still present.
    best_model_path = os.path.join(results_dir, "best_ppo_model.pt")
    if not os.path.exists(best_model_path):
        best_model_path = final_model_path
    try:
        best_actor.load_state_dict(torch.load(best_model_path, map_location=device))
    except RuntimeError:
        if best_model_path == final_model_path:
            raise
        print("Existing best PPO checkpoint is incompatible with the Phase 2 50/70 interface; using final checkpoint.")
        best_actor.load_state_dict(torch.load(final_model_path, map_location=device))
    
    print("Evaluating Best PPO Model (1500 Greedy Games)...")
    best_ppo_results = evaluate_policy(env, best_actor, device, is_random_baseline=False)
    
    print("Evaluating Final PPO Model (1500 Greedy Games)...")
    final_actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    final_actor.load_state_dict(torch.load(final_model_path, map_location=device))
    final_ppo_results = evaluate_policy(env, final_actor, device, is_random_baseline=False)
    
    # 5. Generate Performance Plots
    print("\nGenerating Performance Plots...")
    fig, axs = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle("PPO Performance Analysis - Mini Pokémon TCG RL", fontsize=16, fontweight='bold', y=0.97)
    
    # Pane 1: Training Reward
    axs[0, 0].plot(history["update"], history["avg_episode_reward"], color='lightblue', alpha=0.5, label='Update Avg Reward')
    rolling_r = pd.Series(history["avg_episode_reward"]).rolling(10, min_periods=1).mean()
    axs[0, 0].plot(history["update"], rolling_r, color='royalblue', linewidth=2, label='10-Up Rolling Mean')
    axs[0, 0].set_title("Training Episode Reward vs. Update", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("PPO Update")
    axs[0, 0].set_ylabel("Average Reward")
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()
    
    # Pane 2: Rolling Win Rate
    axs[0, 1].plot(history["update"], history["win_rate"], color='forestgreen', linewidth=2, label='100-Ep Rolling Win Rate')
    axs[0, 1].axhline(y=baseline_win_rate/100.0, color='red', linestyle='--', alpha=0.8, label=f'Random Baseline ({baseline_win_rate:.1f}%)')
    axs[0, 1].set_title("Rolling 100-Episode Win Rate vs. Update", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("PPO Update")
    axs[0, 1].set_ylabel("Win Rate (Fraction)")
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    axs[0, 1].legend()
    
    # Pane 3: Loss curves
    axs[1, 0].plot(history["update"], history["policy_loss"], color='firebrick', label='Policy Loss (Actor)')
    axs[1, 0].set_ylabel("Policy Loss", color='firebrick')
    axs[1, 0].tick_params(axis='y', labelcolor='firebrick')
    
    ax_vf = axs[1, 0].twinx()
    ax_vf.plot(history["update"], history["value_loss"], color='navy', label='Value Loss (Critic)', alpha=0.7)
    ax_vf.set_ylabel("Value Loss", color='navy')
    ax_vf.tick_params(axis='y', labelcolor='navy')
    
    axs[1, 0].set_title("Training Actor (Policy) vs. Critic (Value) Loss", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("PPO Update")
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 4: Entropy / KL diagnostics
    axs[1, 1].plot(history["update"], history["entropy"], color='purple', label='Shannon Entropy')
    axs[1, 1].set_ylabel("Entropy", color='purple')
    axs[1, 1].tick_params(axis='y', labelcolor='purple')
    
    ax_kl = axs[1, 1].twinx()
    ax_kl.plot(history["update"], history["approximate_KL"], color='darkgoldenrod', label='Approximate KL', alpha=0.7)
    ax_kl.set_ylabel("Approximate KL", color='darkgoldenrod')
    ax_kl.tick_params(axis='y', labelcolor='darkgoldenrod')
    
    axs[1, 1].set_title("Policy Entropy & Approximate KL Divergence", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("PPO Update")
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 5: Episode Length
    axs[2, 0].plot(history["update"], history["avg_episode_length"], color='chocolate', linewidth=2)
    axs[2, 0].set_title("Average Episode Pacing (Game Length) vs. Update", fontsize=12, fontweight='bold')
    axs[2, 0].set_xlabel("PPO Update")
    axs[2, 0].set_ylabel("Steps per Game")
    axs[2, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 6: Evaluation Comparison Bar Chart
    categories = ['Random Baseline', 'Best PPO Actor', 'Final PPO Actor']
    win_rates = [
        baseline_win_rate, 
        best_ppo_results["overall"]["win_rate"], 
        final_ppo_results["overall"]["win_rate"]
    ]
    bars = axs[2, 1].bar(categories, win_rates, color=['salmon', 'dodgerblue', 'mediumpurple'], width=0.5)
    axs[2, 1].set_title("Greedy Win Rate Evaluation Comparison", fontsize=12, fontweight='bold')
    axs[2, 1].set_ylabel("Win Rate (%)")
    axs[2, 1].set_ylim(0, 110)
    axs[2, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        axs[2, 1].annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
                            
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plot_path = os.path.join(results_dir, "ppo_performance_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance plots successfully saved to: {plot_path}")
    
    # 6. Print detailed seed range breakdown
    models_data = [
        ("Random Baseline", baseline_results),
        ("Best PPO Actor", best_ppo_results),
        ("Final PPO Actor", final_ppo_results)
    ]
    
    print("\n" + "="*80)
    print("                  SEED RANGE PERFORMANCE BREAKDOWN                    ")
    print("="*80)
    for model_name, res in models_data:
        print(f"\nModel: {model_name}")
        print("-" * 50)
        for range_lbl, data in res["ranges"].items():
            print(f"  Range {range_lbl} | Games: {data['games']} | Wins: {data['wins']} | "
                  f"Losses: {data['losses']} | Win Rate: {data['win_rate']:.1f}% | "
                  f"Avg Reward: {data['avg_reward']:.2f} | Avg Length: {data['avg_length']:.1f}")
                  
    # 7. Print final comparative table
    print("\n" + "="*95)
    print("                              FINAL COMPARATIVE BENCHMARK TABLE                              ")
    print("="*95)
    header = f"{'Model':20} | {'Games':5} | {'Wins':5} | {'Losses':6} | {'Win Rate':8} | {'Avg Reward':10} | {'Avg Length':10}"
    print(header)
    print("-" * 95)
    
    for model_name, res in models_data:
        o = res["overall"]
        row = (f"{model_name:20} | {o['games']:5d} | {o['wins']:5d} | {o['losses']:6d} | "
               f"{o['win_rate']:7.1f}% | {o['avg_reward']:10.2f} | {o['avg_length']:10.1f}")
        print(row)
    print("=" * 95)
    
    # 8. Print final concise comparison
    improvement = best_ppo_results["overall"]["win_rate"] - baseline_win_rate
    print("\n" + "="*70)
    print("                  TCG EXPERIMENT FINAL SUMMARY                        ")
    print("="*70)
    print(f"Device Used:                    {device}")
    print(f"Total Training Time:            {training_time:.1f} seconds")
    print(f"Random baseline:                {baseline_win_rate:.1f}%")
    print(f"Best PPO:                       {best_ppo_results['overall']['win_rate']:.1f}%")
    print(f"Final PPO:                      {final_ppo_results['overall']['win_rate']:.1f}%")
    print(f"Best DQN:                       99.3% (DQN Benchmark reference)")
    print(f"PPO Improvement over Random:    +{improvement:.1f} percentage points")
    print("="*70 + "\n")

if __name__ == "__main__":
    # If called with an environment variable or argument '--test', runs a rapid test run
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    main(is_test_run=test_mode)
