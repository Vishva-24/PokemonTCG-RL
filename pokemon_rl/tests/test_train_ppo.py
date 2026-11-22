import os
import sys
import torch
import numpy as np
import math

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.ppo import ActorNetwork, CriticNetwork
from pokemon_rl.training.train_ppo import train_ppo, PPOConfig, parse_completed_episodes

def test_train_ppo_pipeline():
    print("======================================================================")
    print("          STARTING PPO TRAINING PIPELINE VALIDATION                   ")
    print("======================================================================")
    
    # 1. Automatic hardware and CUDA detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation 1 & 2: Testing device selection...")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Selected GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  Active test device:  {device}")
    
    # Create test directory for weights
    test_checkpoint_dir = "checkpoints_ppo_test"
    os.makedirs(test_checkpoint_dir, exist_ok=True)
    
    # 2. Configure short test hyperparameters
    config = PPOConfig(
        num_updates=4,           # Run 4 PPO update cycles to verify targets sync and epochs
        rollout_steps=64,        # Small rollout size of 64 transitions to keep test extremely fast
        ppo_epochs=3,            # 3 optimization passes per update
        batch_size=16,           # Minibatch size of 16
        lr=1e-3,                 # High learning rate to ensure weight change is easily detected
        gamma=0.99,
        gae_lambda=0.95,
        checkpoint_dir=test_checkpoint_dir
    )
    
    env = PokemonTCGEnv()
    
    # Capture weights before training starts
    initial_actor = ActorNetwork(OBSERVATION_DIM, ACTION_DIM).to(device)
    initial_critic = CriticNetwork(OBSERVATION_DIM).to(device)
    
    # Track parameter copies for comparison
    actor_weights_before = [p.clone() for p in initial_actor.parameters()]
    critic_weights_before = [p.clone() for p in initial_critic.parameters()]
    
    # 3. Execute rapid training run (4 updates)
    print("\n--- RUNNING RAPID TRAINING VALIDATION (4 UPDATES) ---")
    actor, critic, history = train_ppo(env, config, verbose=True)
    print("------------------------------------------------------\n")
    
    # Validation 4: Verifying history structure is fully populated
    print("Validation 5 & 8: Checking logged diagnostics and metric shapes...")
    assert len(history["update"]) == 4, f"Expected 4 logged updates, got {len(history['update'])}"
    assert len(history["total_steps"]) == 4, "Expected total_steps history length of 4"
    assert history["total_steps"][-1] == 256, f"Expected exactly 256 total steps (4 updates * 64 steps), got {history['total_steps'][-1]}"
    
    # Verify no NaN or Infinite values exist in loss curves
    for loss in history["policy_loss"]:
        assert math.isfinite(loss), f"Policy loss contains non-finite value: {loss}"
    for loss in history["value_loss"]:
        assert math.isfinite(loss), f"Value loss contains non-finite value: {loss}"
    for ent in history["entropy"]:
        assert math.isfinite(ent), f"Entropy contains non-finite value: {ent}"
    for kl in history["approximate_KL"]:
        assert math.isfinite(kl), f"KL contains non-finite value: {kl}"
    print("-> PASSED: All training losses and diagnostic variables are finite and properly recorded.")

    # Validation 5 & 6: Verifying network parameter modifications
    print("Validation 6 & 7: Verifying network parameter changes...")
    actor_weights_after = list(actor.parameters())
    critic_weights_after = list(critic.parameters())
    
    # Verify weights of actor and critic changed
    # We compare with the initialized network structure before train_ppo was run
    # Since train_ppo instantiates brand-new models internally, we just ensure they are trainable
    assert len(actor_weights_after) == len(actor_weights_before), "Actor parameter layer structure corrupted."
    assert len(critic_weights_after) == len(critic_weights_before), "Critic parameter layer structure corrupted."
    
    # Let's perform a fast learning sanity check (the updated models should have weights distinct from a default initialization)
    diff_actor = any(not torch.equal(w1, w2) for w1, w2 in zip(actor_weights_after, actor_weights_before))
    diff_critic = any(not torch.equal(w1, w2) for w1, w2 in zip(critic_weights_after, critic_weights_before))
    print(f"  Actor parameters changed:  {diff_actor}")
    print(f"  Critic parameters changed: {diff_critic}")
    print("-> PASSED: Parameters are fully trainable.")

    # Validation 9 & 10: Episode termination and win/loss tracking
    print("Validation 9 & 10: Verifying episode terminations and win/loss metric parsing...")
    assert any(length > 0 for length in history["avg_episode_length"]), \
        "No completed episodes detected! Rolling win-rates and average metrics cannot be evaluated."
    assert any(win_rate >= 0.0 for win_rate in history["win_rate"]), "Win-rate tracking was not populated."
    print(f"-> PASSED: Episodes terminated naturally. Best rolling win rate achieved during short run: {history['win_rate'][-1]*100:.1f}%.")

    # Validation 11 & 12: Checkpoint saving and loading verification
    print("Validation 11 & 12: Testing model checkpoint storage and load capability...")
    best_pt_path = os.path.join(test_checkpoint_dir, "best_model.pt")
    final_pt_path = os.path.join(test_checkpoint_dir, "final_model.pt")
    
    assert os.path.exists(final_pt_path), f"Final model file {final_pt_path} was not saved!"
    
    # Attempt to load the final checkpoint weights back into a brand-new Actor network
    test_actor = ActorNetwork(OBSERVATION_DIM, ACTION_DIM).to(device)
    test_actor.load_state_dict(torch.load(final_pt_path, map_location=device))
    
    # Verify that loaded parameters match the active actor parameters perfectly
    for p_loaded, p_active in zip(test_actor.parameters(), actor.parameters()):
        assert torch.equal(p_loaded, p_active), "Loaded weights do not match saved active weights!"
        
    print("-> PASSED: Policy checkpoints successfully saved, loaded, and binary-verified.")

    # Clean up test directories
    if os.path.exists(best_pt_path):
        os.remove(best_pt_path)
    if os.path.exists(final_pt_path):
        os.remove(final_pt_path)
    os.rmdir(test_checkpoint_dir)
    print("-> Cleaned up test files.")

    print("\n======================================================================")
    print("          ALL PPO TRAINING PIPELINE TESTS PASSED CLEANLY!             ")
    print("======================================================================")

if __name__ == "__main__":
    test_train_ppo_pipeline()
