import os
import torch
import numpy as np
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.dqn import DQNetwork
from pokemon_rl.training.train_dqn import train_dqn, DQNConfig

def test_dqn_training_pipeline():
    print("======================================================================")
    print("          STARTING DQN TRAINING LOOP PIPELINE VALIDATION              ")
    print("======================================================================")

    # 1. Initialize environment
    env = PokemonTCGEnv()
    
    # Use hyper-conservative parameters for a fast 10-episode validation run
    # Set batch size small and buffer capacity small so training steps start early
    config = DQNConfig(
        num_episodes=10,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=1000,
        batch_size=8,
        epsilon_start=1.0,
        epsilon_decay=0.90,  # Fast decay to verify epsilon decreases quickly
        epsilon_min=0.05,
        target_update_freq=3,  # Frequent target updates for testing
        max_steps_per_episode=150,
        checkpoint_dir="/workspace/scratch/checkpoints_test"
    )

    print(f"Hyperparameters chosen for rapid verification:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    print()

    # 2. Run the training loop
    print("--- RUNNING RAPID TRAINING VALIDATION (10 EPISODES) ---")
    trainer, history = train_dqn(env, config, verbose=True)
    print("------------------------------------------------------\n")

    # 3. Assertions and Verifications
    
    # Verification 1: Training runs without crashing & completes 10 episodes
    assert len(history["episode_rewards"]) == 10, "Should have processed 10 episodes."
    print("Validation 1: Training loop executed completely without crashing.")

    # Verification 2: Episodes terminate and lengths are recorded
    assert all(length > 0 for length in history["episode_lengths"]), "Episode lengths must be positive."
    print("Validation 2: Episodes terminated naturally with correct steps recorded.")

    # Verification 3: Replay buffer receives transitions
    buffer_len = len(trainer.replay_buffer)
    assert buffer_len > 0, "Replay buffer should have stored transitions."
    print(f"Validation 3: Replay buffer received and stored {buffer_len} transitions.")

    # Verification 4: train_step() is called when buffer size exceeds batch size (8)
    # The first few steps should not train, but once buffer length >= 8, it should start training.
    # At 10 episodes, total transitions should easily be > 8, so there should be recorded losses.
    training_steps_occurred = any(loss > 0.0 for loss in history["losses"])
    assert training_steps_occurred, "Loss should be recorded once buffer length >= batch size."
    print("Validation 4: train_step() was successfully called and executed when buffer was ready.")

    # Verification 5: Loss values are recorded (either 0.0 before batch size, or real float)
    assert len(history["losses"]) == 10, "Loss history must have length equal to number of episodes."
    print("Validation 5: Episode average loss values are correctly recorded.")

    # Verification 6: Epsilon decreases from its starting value (1.0)
    assert history["epsilons"][-1] < 1.0, "Epsilon must decrease due to decay parameter."
    print(f"Validation 6: Epsilon decreased successfully from 1.0 to {history['epsilons'][-1]:.4f}.")

    # Verification 7: Episode rewards are recorded
    assert len(history["episode_rewards"]) == 10, "Reward history must contain 10 elements."
    print("Validation 7: Episode cumulative rewards are correctly recorded.")

    # Verification 8: Win/loss statistics are recorded (0.0 or 1.0)
    assert len(history["results"]) == 10, "Winner/loser records must contain 10 elements."
    assert all(r in [0.0, 1.0] for r in history["results"]), "Results must be either 1.0 (win) or 0.0 (loss)."
    print("Validation 8: Match outcomes (win/loss records) are correctly tracked.")

    # Verification 9: Final model checkpoint can be saved and loaded
    checkpoint_file = os.path.join(config.checkpoint_dir, "final_model.pt")
    assert os.path.exists(checkpoint_file), f"Final model file {checkpoint_file} not found."
    
    # Initialize a clean neural network and load the state dict
    clean_net = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)
    clean_net.load_state_dict(torch.load(checkpoint_file))
    
    # Verify parameter loading matches the policy network's weights
    for p_clean, p_trained in zip(clean_net.parameters(), trainer.policy_net.parameters()):
        assert torch.equal(p_clean, p_trained), "Loaded parameters do not match policy network."
    print("Validation 9: Model weights successfully saved, loaded, and verified.")

    # Verification 10: Works on CPU
    assert next(trainer.policy_net.parameters()).device.type == "cpu", "Trainer model should be executing on CPU."
    print("Validation 10: All training, updates, and evaluations successfully executed on CPU.")

    print("\n======================================================================")
    print("          ALL DQN TRAINING PIPELINE VALIDATIONS PASSED CLEANLY!       ")
    print("======================================================================")

if __name__ == "__main__":
    test_dqn_training_pipeline()
