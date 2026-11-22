import os
import sys
import torch
import numpy as np

# Ensure local 'pokemon_rl' package is discoverable from the script's root path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.dqn import select_action, DQNetwork

# Evaluation Configurations
SEED_RANGES = [
    (10000, 10499),
    (20000, 20499),
    (30000, 30499)
]
MAX_STEPS_PER_EPISODE = 200

def evaluate_policy(env: PokemonTCGEnv, model: torch.nn.Module, device: torch.device, is_random_baseline: bool = False) -> dict:
    """Evaluates a policy across the three configured seed ranges.
    
    Args:
        env: The Gymnasium Pokémon environment.
        model: The DQNetwork model to evaluate.
        device: Torch device (CPU/CUDA).
        is_random_baseline: If True, uses random action selection.
        
    Returns:
        A dictionary containing breakdown metrics per range and overall metrics.
    """
    model.eval()
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
                    # Epsilon = 1.0 forces select_action to pick a random legal move
                    action_idx = select_action(obs, model, mask, epsilon=1.0, device=device)
                else:
                    # Epsilon = 0.0 forces greedy legal action selection
                    action_idx = select_action(obs, model, mask, epsilon=0.0, device=device)
                    
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

def main():
    print("======================================================================")
    # Automatically detect hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"AUTOMATIC HARDWARE DETECTION:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device:     {torch.cuda.get_device_name(0)}")
    print(f"Active Device:   {device}")
    print("======================================================================\n")
    
    env = PokemonTCGEnv()
    
    # 1. Random Baseline
    print("Evaluating Random Baseline...")
    dummy_model = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    baseline_results = evaluate_policy(env, dummy_model, device, is_random_baseline=True)
    
    # Paths for model checkpoints relative to script directory
    results_dir = os.path.join(script_dir, "results")
    best_model_path = os.path.join(results_dir, "best_model.pt")
    final_model_path = os.path.join(results_dir, "final_model.pt")
    
    # Helper to check if model files exist
    if not os.path.exists(best_model_path) or not os.path.exists(final_model_path):
        print("ERROR: Could not find model files in the 'results' directory!")
        print(f"Checked paths:\n  - {best_model_path}\n  - {final_model_path}")
        print("Please ensure your training run completed successfully and saved the weights.")
        sys.exit(1)
        
    # 2. Best DQN Model
    print("Evaluating Best DQN Model...")
    best_model = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_results = evaluate_policy(env, best_model, device, is_random_baseline=False)
    
    # 3. Final DQN Model
    print("Evaluating Final DQN Model...")
    final_model = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    final_model.load_state_dict(torch.load(final_model_path, map_location=device))
    final_results = evaluate_policy(env, final_model, device, is_random_baseline=False)
    
    # 4. Print detailed results per seed range
    models_data = [
        ("Random Baseline", baseline_results),
        ("Best DQN Model", best_results),
        ("Final DQN Model", final_results)
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
                  
    # 5. Print final comparative table
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
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
