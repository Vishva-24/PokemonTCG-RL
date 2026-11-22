import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

# Ensure local 'pokemon_rl' package is discoverable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.lstm_ppo import LSTMActorNetwork, LSTMCriticNetwork, MaskedCategorical, LSTMPPORolloutBuffer
from pokemon_rl.agents.lstm_ppo_trainer import LSTMPPOTrainer, compute_recurrent_gae_and_returns
from pokemon_rl.training.train_lstm_ppo import train_lstm_ppo, LSTMPPOConfig

# Recurrent Experiment Parameters
NUM_UPDATES = 500
ROLLOUT_STEPS = 2048
SEQUENCE_LENGTH = 16
BATCH_SIZE = 4
PPO_EPOCHS = 10
LEARNING_RATE = 5e-5
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
    print("            DETECTING RECURRENT HARDWARE & PYTORCH SYSTEM             ")
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

def evaluate_recurrent_policy(
    env: PokemonTCGEnv,
    actor: LSTMActorNetwork,
    device: torch.device,
    is_random_baseline: bool = False
) -> dict:
    """Evaluates a recurrent policy across the three configured seed ranges of 500 games each.
    
    Resets the hidden states at the beginning of each game to prevent temporal leakage.
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
            
            # Recurrent hidden states must be initialized cleanly at start of each game
            h_actor, c_actor = actor.init_hidden(batch_size=1, device=device)
            
            while not (terminated or truncated) and steps < MAX_STEPS_PER_EPISODE:
                mask = env.get_action_mask()
                
                if is_random_baseline:
                    legal_indices = np.where(mask == 1.0)[0]
                    action_idx = int(np.random.choice(legal_indices))
                else:
                    # Greedy evaluation over probability outputs
                    state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        logits, (h_actor, c_actor) = actor(state_t, (h_actor, c_actor))
                        policy = MaskedCategorical(logits, mask_t)
                        probs = policy.probs
                        action_idx = int(torch.argmax(probs).item())
                        
                obs, reward, terminated, truncated, _ = env.step(action_idx)
                game_reward += reward
                steps += 1
                
            winner = env.state.winner_idx
            if winner == 0:
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
        
        # Accumulate
        results["overall"]["games"] += num_games
        results["overall"]["wins"] += wins
        results["overall"]["losses"] += losses
        results["overall"]["total_reward"] += total_reward
        results["overall"]["total_steps"] += total_steps
        
    o = results["overall"]
    o["win_rate"] = (o["wins"] / o["games"]) * 100.0
    o["avg_reward"] = o["total_reward"] / o["games"]
    o["avg_length"] = o["total_steps"] / o["games"]
    
    return results

def main(is_test_run: bool = False):
    print_hardware_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for experiment: {device}\n")
    
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    num_updates = 5 if is_test_run else NUM_UPDATES
    rollout_steps = 64 if is_test_run else ROLLOUT_STEPS
    batch_size = 2 if is_test_run else BATCH_SIZE
    
    global SEED_RANGES
    if is_test_run:
        SEED_RANGES = [(10000, 10004), (20000, 20004), (30000, 30004)]
        
    env = PokemonTCGEnv()
    
    # 1. Evaluate Untrained / Random Baseline
    print("Evaluating Untrained/Random Baseline (1500 Games)...")
    dummy_actor = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    baseline_results = evaluate_recurrent_policy(env, dummy_actor, device, is_random_baseline=True)
    baseline_win_rate = baseline_results["overall"]["win_rate"]
    baseline_avg_reward = baseline_results["overall"]["avg_reward"]
    baseline_avg_len = baseline_results["overall"]["avg_length"]
    
    print(f"Baseline Random Win Rate: {baseline_win_rate:.1f}%")
    print(f"Baseline Average Reward:  {baseline_avg_reward:.2f}")
    print(f"Baseline Average Length:  {baseline_avg_len:.1f} steps\n")
    
    # 2. Run LSTM-PPO Training
    config = LSTMPPOConfig(
        num_updates=num_updates,
        rollout_steps=rollout_steps,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=batch_size,
        ppo_epochs=PPO_EPOCHS,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        entropy_coefficient=ENTROPY_COEF,
        value_loss_coefficient=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        checkpoint_dir=results_dir
    )
    
    print(f"Training LSTM-PPO on {device} for {num_updates} updates...")
    start_time = time.time()
    actor, critic, history = train_lstm_ppo(env, config, verbose=True)
    training_time = time.time() - start_time
    print(f"\nTraining Complete! Time taken: {training_time:.1f} seconds.")
    
    # Save training metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_csv_path = os.path.join(results_dir, "lstm_ppo_training_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Recurrent metrics successfully saved to: {metrics_csv_path}")
    
    # 3. Evaluate Checkpointed recurrent models
    print("\nLoading Best LSTM-PPO checkpoint for Greedy Evaluation...")
    best_actor = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    final_model_path = os.path.join(results_dir, "final_lstm_ppo_actor.pt")
    best_model_path = os.path.join(results_dir, "best_lstm_ppo_actor.pt")
    if not os.path.exists(best_model_path):
        best_model_path = final_model_path
    try:
        best_actor.load_state_dict(torch.load(best_model_path, map_location=device))
    except RuntimeError:
        if best_model_path == final_model_path:
            raise
        print("Existing best LSTM-PPO checkpoint is incompatible with the Phase 2 50/70 interface; using final checkpoint.")
        best_actor.load_state_dict(torch.load(final_model_path, map_location=device))
    
    print("Evaluating Best LSTM-PPO Model (1500 Greedy Games)...")
    best_recurrent_results = evaluate_recurrent_policy(env, best_actor, device, is_random_baseline=False)
    
    print("Evaluating Final LSTM-PPO Model (1500 Greedy Games)...")
    final_actor = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    final_actor.load_state_dict(torch.load(final_model_path, map_location=device))
    final_recurrent_results = evaluate_recurrent_policy(env, final_actor, device, is_random_baseline=False)
    
    # 4. Generate high-resolution plots dashboard
    print("\nGenerating Recurrent Performance Plots Dashboard...")
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("LSTM-PPO Performance & Diagnostics - Mini Pokémon TCG RL", fontsize=16, fontweight='bold', y=0.97)
    
    # Pane 1: Reward
    axs[0, 0].plot(history["update"], history["avg_episode_reward"], color='lightblue', alpha=0.4, label='Update Reward')
    rolling_r = pd.Series(history["avg_episode_reward"]).rolling(10, min_periods=1).mean()
    axs[0, 0].plot(history["update"], rolling_r, color='royalblue', linewidth=2, label='10-Up Rolling Mean')
    axs[0, 0].set_title("Training Episode Reward vs. Update", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("PPO Update")
    axs[0, 0].set_ylabel("Average Reward")
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()
    
    # Pane 2: Win Rate
    axs[0, 1].plot(history["update"], history["win_rate"], color='forestgreen', linewidth=2, label='100-Ep Rolling Win Rate')
    axs[0, 1].axhline(y=baseline_win_rate/100.0, color='red', linestyle='--', alpha=0.8, label=f'Random ({baseline_win_rate:.1f}%)')
    axs[0, 1].set_title("Rolling 100-Episode Win Rate vs. Update", fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel("PPO Update")
    axs[0, 1].set_ylabel("Win Rate (Fraction)")
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    axs[0, 1].legend()
    
    # Pane 3: Policy Loss
    axs[0, 2].plot(history["update"], history["policy_loss"], color='firebrick')
    axs[0, 2].set_title("Actor Policy Loss vs. Update", fontsize=11, fontweight='bold')
    axs[0, 2].set_xlabel("PPO Update")
    axs[0, 2].set_ylabel("Policy Loss")
    axs[0, 2].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 4: Value Loss
    axs[1, 0].plot(history["update"], history["value_loss"], color='navy')
    axs[1, 0].set_title("Critic Value Loss vs. Update", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("PPO Update")
    axs[1, 0].set_ylabel("Value Loss (MSE)")
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 5: Entropy
    axs[1, 1].plot(history["update"], history["entropy"], color='purple')
    axs[1, 1].set_title("Shannon Policy Entropy vs. Update", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("PPO Update")
    axs[1, 1].set_ylabel("Entropy")
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 6: Approximate KL
    axs[1, 2].plot(history["update"], history["approximate_KL"], color='darkgoldenrod')
    axs[1, 2].set_title("Approximate KL Divergence", fontsize=11, fontweight='bold')
    axs[1, 2].set_xlabel("PPO Update")
    axs[1, 2].set_ylabel("KL Value")
    axs[1, 2].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 7: Clip Fraction
    axs[2, 0].plot(history["update"], history["clip_fraction"], color='teal')
    axs[2, 0].set_title("Surrogate Objective Clip Fraction", fontsize=11, fontweight='bold')
    axs[2, 0].set_xlabel("PPO Update")
    axs[2, 0].set_ylabel("Clip Fraction")
    axs[2, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 8: Episode Length (Game pacing)
    axs[2, 1].plot(history["update"], history["avg_episode_length"], color='chocolate', linewidth=2)
    axs[2, 1].set_title("Average Game Length pacing vs. Update", fontsize=11, fontweight='bold')
    axs[2, 1].set_xlabel("PPO Update")
    axs[2, 1].set_ylabel("Steps per Game")
    axs[2, 1].grid(True, linestyle=':', alpha=0.6)
    
    # Pane 9: Evaluation Comparison
    categories = ['Random', 'Best LSTM-PPO', 'Final LSTM-PPO']
    win_rates = [
        baseline_win_rate,
        best_recurrent_results["overall"]["win_rate"],
        final_recurrent_results["overall"]["win_rate"]
    ]
    bars = axs[2, 2].bar(categories, win_rates, color=['salmon', 'dodgerblue', 'mediumpurple'], width=0.4)
    axs[2, 2].set_title("Greedy Win Rate Evaluation (1500 Games)", fontsize=11, fontweight='bold')
    axs[2, 2].set_ylabel("Win Rate (%)")
    axs[2, 2].set_ylim(0, 110)
    axs[2, 2].grid(True, axis='y', linestyle=':', alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        axs[2, 2].annotate(f'{h:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
                            
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plot_path = os.path.join(results_dir, "lstm_ppo_performance_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots successfully saved to: {plot_path}")
    
    # Print breakdown
    models_data = [
        ("Random Baseline", baseline_results),
        ("Best LSTM-PPO Actor", best_recurrent_results),
        ("Final LSTM-PPO Actor", final_recurrent_results)
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
                  
    # Final comparison summary table
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
    print("="*95)
    
    improvement = best_recurrent_results["overall"]["win_rate"] - baseline_win_rate
    print("\n" + "="*70)
    print("                  TCG EXPERIMENT FINAL SUMMARY                        ")
    print("="*70)
    print(f"Device Used:                    {device}")
    print(f"Total Training Time:            {training_time:.1f} seconds")
    print(f"Random baseline:                {baseline_win_rate:.1f}%")
    print(f"Best LSTM-PPO:                  {best_recurrent_results['overall']['win_rate']:.1f}%")
    print(f"Final LSTM-PPO:                 {final_recurrent_results['overall']['win_rate']:.1f}%")
    print(f"Best DQN:                       99.3% (DQN Benchmark reference)")
    print(f"Final PPO:                      99.5% (PPO Benchmark reference)")
    print(f"LSTM-PPO Improvement over Random: +{improvement:.1f} percentage points")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    main(is_test_run=test_mode)
