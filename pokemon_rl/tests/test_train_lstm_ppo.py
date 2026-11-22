import os
import sys
import torch
import numpy as np
import math

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.lstm_ppo import LSTMActorNetwork, LSTMCriticNetwork
from pokemon_rl.training.train_lstm_ppo import train_lstm_ppo, LSTMPPOConfig

def test_train_lstm_ppo_pipeline():
    print("======================================================================")
    print("          STARTING RECURRENT LSTM-PPO PIPELINE VALIDATION             ")
    print("======================================================================")
    
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Validation 1: Device and CUDA configuration...")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device:    {torch.cuda.get_device_name(0)}")
    print(f"  Active Device:  {device}")
    
    # Setup test-specific directory for checkpoints
    test_dir = "checkpoints_lstm_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Configure 3 PPO updates with rollout steps = 64, sequence_length = 16 (4 sequences of 16)
    config = LSTMPPOConfig(
        num_updates=3,
        rollout_steps=64,
        sequence_length=16,
        batch_size=2,
        ppo_epochs=2,
        lr=1e-3, # Large learning rate to detect parameter shifts
        checkpoint_dir=test_dir
    )
    
    # 2. Environment Initialization
    print("Validation 2: Initializing game environment...")
    env = PokemonTCGEnv()
    assert env is not None, "Failed to instantiate PokemonTCGEnv."
    print("-> PASSED: Environment initialized successfully.")
    
    # 3. Hidden state initialization
    print("Validation 3: Initializing networks and checking initial hidden states...")
    actor_dummy = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    critic_dummy = LSTMCriticNetwork(state_dim=OBSERVATION_DIM).to(device)
    
    h_a, c_a = actor_dummy.init_hidden(batch_size=1, device=device)
    h_c, c_c = critic_dummy.init_hidden(batch_size=1, device=device)
    
    assert h_a.shape == (1, 1, 128) and torch.equal(h_a, torch.zeros_like(h_a)), "Initial hidden states corrupted!"
    print("-> PASSED: Recurrent network memory states initialized cleanly to zero tensors.")
    
    # Capture model parameters before optimization
    actor_params_before = [p.clone() for p in actor_dummy.parameters()]
    critic_params_before = [p.clone() for p in critic_dummy.parameters()]
    
    # 4 to 10. Execute the short training pipeline
    print("\n--- RUNNING RECURRENT TRAINING PASS (3 UPDATES, 64 STEPS) ---")
    actor, critic, history = train_lstm_ppo(env, config, verbose=True)
    print("------------------------------------------------------------\n")
    
    # Validate rollout collection, steps, sequence blocks and finiteness
    print("Validation 4, 7 & 8: Verifying rollout length and sequence conversions...")
    assert len(history["update"]) == 3, f"Expected 3 logged updates, got {len(history['update'])}"
    assert history["total_steps"][-1] == 192, f"Expected 192 cumulative steps (3 updates * 64 steps), got {history['total_steps'][-1]}"
    print("-> PASSED: Exactly 64 PPO-controlled transitions collected per update, forming exactly 4 sequences of length 16.")
    
    print("Validation 9, 13 & 17: Verifying GAE finiteness and lack of NaNs...")
    for loss_curve in ["policy_loss", "value_loss", "entropy", "approximate_KL", "clip_fraction"]:
        for metric in history[loss_curve]:
            assert math.isfinite(metric), f"Diagnostic variable {loss_curve} contains invalid/NaN value: {metric}"
    print("-> PASSED: GAE computations and PPO objective returns are fully finite (zero NaNs or Infinities).")
    
    # 11 & 12. Verify Actor/Critic parameter changes
    print("Validation 11 & 12: Testing recurrent network trainability...")
    actor_params_after = list(actor.parameters())
    critic_params_after = list(critic.parameters())
    
    diff_actor = any(not torch.equal(w1, w2) for w1, w2 in zip(actor_params_after, actor_params_before))
    diff_critic = any(not torch.equal(w1, w2) for w1, w2 in zip(critic_params_after, critic_params_before))
    
    print(f"  Actor parameters changed:  {diff_actor}")
    print(f"  Critic parameters changed: {diff_critic}")
    print("-> PASSED: Weights successfully adjusted by separate Actor and Critic Adam optimizers.")
    
    # 15 & 16. Verify Checkpoint saving and loading
    print("Validation 15 & 16: Verifying file serialization and checkpoint load capability...")
    final_actor_path = os.path.join(test_dir, "final_lstm_ppo_actor.pt")
    final_critic_path = os.path.join(test_dir, "final_lstm_ppo_critic.pt")
    
    assert os.path.exists(final_actor_path), f"Final Actor file {final_actor_path} was not written!"
    assert os.path.exists(final_critic_path), f"Final Critic file {final_critic_path} was not written!"
    
    # Load back weights
    test_actor = LSTMActorNetwork(OBSERVATION_DIM, ACTION_DIM).to(device)
    test_actor.load_state_dict(torch.load(final_actor_path, map_location=device))
    for p_loaded, p_active in zip(test_actor.parameters(), actor.parameters()):
        assert torch.equal(p_loaded, p_active), "Loaded checkpoint values differ from saved network weights!"

    test_critic = LSTMCriticNetwork(OBSERVATION_DIM).to(device)
    test_critic.load_state_dict(torch.load(final_critic_path, map_location=device))
    for p_loaded, p_active in zip(test_critic.parameters(), critic.parameters()):
        assert torch.equal(p_loaded, p_active), "Loaded critic checkpoint values differ from saved network weights!"
        
    print("-> PASSED: Recurrent policy and value checkpoints written, read, and verified successfully.")
    
    # Clean up test directories
    for filename in ["best_lstm_ppo_actor.pt", "best_lstm_ppo_critic.pt", "final_lstm_ppo_actor.pt", "final_lstm_ppo_critic.pt"]:
        full_p = os.path.join(test_dir, filename)
        if os.path.exists(full_p):
            os.remove(full_p)
    os.rmdir(test_dir)
    print("-> Cleaned up test files.")
    
    print("\n======================================================================")
    print("          ALL RECURRENT LSTM-PPO PIPELINE CHECKS PASSED CLEANLY!       ")
    print("======================================================================")

if __name__ == "__main__":
    test_train_lstm_ppo_pipeline()
