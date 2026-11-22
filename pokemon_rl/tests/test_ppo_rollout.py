import torch
import numpy as np
import math

from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.agents.ppo import ActorNetwork, CriticNetwork, PPORolloutBuffer
from pokemon_rl.agents.ppo_rollout import collect_rollout, compute_gae_and_returns

def test_ppo_rollout_full():
    print("======================================================================")
    print("          STARTING PPO ROLLOUT & GAE VALIDATION SUITE                 ")
    print("======================================================================")
    
    # Detect hardware automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device for Rollout Test: {device}")
    
    # -------------------------------------------------------------------------
    # Initialize Environment and Networks
    # -------------------------------------------------------------------------
    env = PokemonTCGEnv()
    actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    critic = CriticNetwork(state_dim=OBSERVATION_DIM).to(device)
    buffer = PPORolloutBuffer()
    
    # -------------------------------------------------------------------------
    # 1, 2 & 3. Collect rollout of exactly 32 steps
    # -------------------------------------------------------------------------
    print("Validation 1, 2 & 3: Collecting small 32-step rollout from real environment...")
    next_obs, next_info, next_seed, last_value = collect_rollout(
        env=env,
        actor=actor,
        critic=critic,
        buffer=buffer,
        rollout_steps=32,
        device=device,
        seed_sequence=5000
    )
    
    assert buffer.length() == 32, f"Expected exactly 32 transitions, got {buffer.length()}"
    print(f"-> PASSED: Collected {buffer.length()} transitions successfully. Last next_value estimation: {last_value:.4f}")
    
    # Check that every sampled action is strictly legal according to the stored mask
    print("Validation 2: Verifying that all sampled actions are legal...")
    for idx in range(32):
        act = buffer.actions[idx]
        mask = buffer.action_masks[idx]
        assert mask[act] == 1.0, f"At step {idx}, sampled illegal action index {act} under mask {mask}"
    print("-> PASSED: Every sampled action was strictly legal.")

    # -------------------------------------------------------------------------
    # 4 to 9. Convert to Tensors and verify shapes and finiteness
    # -------------------------------------------------------------------------
    print("Validation 4 to 9: Verifying rollout tensor shapes and numerical finiteness...")
    states, actions, rewards, dones, values, log_probs, masks = buffer.get_tensors(device)
    
    assert states.shape == (32, OBSERVATION_DIM), f"Expected states shape (32, {OBSERVATION_DIM}), got {states.shape}"
    assert actions.shape == (32,), f"Expected actions shape (32,), got {actions.shape}"
    assert rewards.shape == (32,), f"Expected rewards shape (32,), got {rewards.shape}"
    assert dones.shape == (32,), f"Expected dones shape (32,), got {dones.shape}"
    assert values.shape == (32,), f"Expected values shape (32,), got {values.shape}"
    assert log_probs.shape == (32,), f"Expected log_probs shape (32,), got {log_probs.shape}"
    assert masks.shape == (32, ACTION_DIM), f"Expected masks shape (32, {ACTION_DIM}), got {masks.shape}"
    
    # Check numerical finiteness
    for idx, log_p in enumerate(buffer.log_probs):
        assert math.isfinite(log_p), f"Log-probability at step {idx} is non-finite: {log_p}"
    for idx, v_val in enumerate(buffer.values):
        assert math.isfinite(v_val), f"Critic value at step {idx} is non-finite: {v_val}"
        
    print("-> PASSED: Rollout tensor shapes are correct and values are fully finite.")

    # -------------------------------------------------------------------------
    # 10, 11 & 12. GAE advantage computation, shapes, and normalization
    # -------------------------------------------------------------------------
    print("Validation 10, 11 & 12: Testing GAE advantage calculation and normalization...")
    rewards_np = rewards.cpu().numpy()
    dones_np = dones.cpu().numpy()
    values_np = values.cpu().numpy()
    
    advantages, returns = compute_gae_and_returns(
        rewards=rewards_np,
        dones=dones_np,
        values=values_np,
        last_value=last_value,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True
    )
    
    assert advantages.shape == (32,), f"Expected advantages shape (32,), got {advantages.shape}"
    assert returns.shape == (32,), f"Expected returns shape (32,), got {returns.shape}"
    
    # Check advantage normalization stats (mean ~0, std ~1)
    mean_adv = advantages.mean()
    std_adv = advantages.std()
    print(f"  Normalized advantages: Mean={mean_adv:.6f}, Std={std_adv:.6f}")
    assert abs(mean_adv) < 1e-4, f"Normalized advantages mean is not approx 0, got {mean_adv}"
    assert abs(std_adv - 1.0) < 1e-4, f"Normalized advantages std is not approx 1, got {std_adv}"
    
    # Clear buffer and verify length
    buffer.clear()
    assert buffer.length() == 0, f"Buffer clear failed to reset size to 0, got {buffer.length()}"
    print("-> PASSED: GAE computation and advantage standardization function perfectly.")

    # -------------------------------------------------------------------------
    # 13. Terminal transition value-bootstrapping bypass test (Synthetic Rollout)
    # -------------------------------------------------------------------------
    print("Validation 13: Verifying that terminal transitions bypass next-state value bootstrapping...")
    # Setup a 5-step synthetic rollout
    synthetic_rewards = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) # Terminal win (+1.0 reward)
    synthetic_dones = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)   # Last step is terminal
    synthetic_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    
    # Let's set a massive next-state value (last_value) which represents a huge bootstrap value
    huge_last_value = 9999.0
    
    # GAE without normalization
    synth_adv, synth_returns = compute_gae_and_returns(
        rewards=synthetic_rewards,
        dones=synthetic_dones,
        values=synthetic_values,
        last_value=huge_last_value,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False
    )
    
    # The last transition is terminal (done = 1.0), so TD residual delta_T should be:
    # delta_T = rewards[T] + gamma * (1 - done_T) * huge_last_value - values[T]
    #         = 1.0 + 0.99 * 0.0 * 9999.0 - 0.5 = 0.5
    # Since done_T is True, last_gae is also reset:
    # advantage_T = delta_T + gamma * lambda * (1 - done_T) * A_{T+1} = delta_T = 0.5
    # return_T = advantage_T + value_T = 0.5 + 0.5 = 1.0 (exact terminal reward, no bootstrapping)
    
    assert abs(synth_returns[-1] - 1.0) < 1e-5, f"Expected final target return of 1.0, but got {synth_returns[-1]}"
    print("-> PASSED: Terminal boundary test confirmed. Terminal target returns strictly ignore next-state value estimations.")

    # -------------------------------------------------------------------------
    # 14 & 15. CPU and CUDA safety check
    # -------------------------------------------------------------------------
    print("Validation 14 & 15: Verifying CPU/CUDA cross-device compatibility...")
    cpu_device = torch.device("cpu")
    actor_cpu = ActorNetwork().to(cpu_device)
    critic_cpu = CriticNetwork().to(cpu_device)
    buffer_cpu = PPORolloutBuffer()
    
    _ = collect_rollout(
        env=env,
        actor=actor_cpu,
        critic=critic_cpu,
        buffer=buffer_cpu,
        rollout_steps=5,
        device=cpu_device,
        seed_sequence=6000
    )
    assert buffer_cpu.length() == 5, "CPU rollout collection failed to collect requested transitions."
    print("-> PASSED: CPU execution verified as stable.")
    
    if torch.cuda.is_available():
        print("CUDA available. Testing CUDA integration...")
        cuda_device = torch.device("cuda")
        actor_cuda = ActorNetwork().to(cuda_device)
        critic_cuda = CriticNetwork().to(cuda_device)
        buffer_cuda = PPORolloutBuffer()
        _ = collect_rollout(
            env=env,
            actor=actor_cuda,
            critic=critic_cuda,
            buffer=buffer_cuda,
            rollout_steps=5,
            device=cuda_device,
            seed_sequence=7000
        )
        assert buffer_cuda.length() == 5
        print("-> PASSED: CUDA execution verified as stable.")
    else:
        print("-> INFO: CUDA GPU not available on this environment (Skipping CUDA test).")

    print("\n======================================================================")
    print("          ALL PPO ROLLOUT VALIDATION TESTS PASSED CLEANLY!            ")
    print("======================================================================")


if __name__ == "__main__":
    test_ppo_rollout_full()
