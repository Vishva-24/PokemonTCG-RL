import torch
import numpy as np
import math
from pokemon_rl.agents.ppo import ActorNetwork, CriticNetwork, MaskedCategorical, PPORolloutBuffer
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def test_ppo_components():
    print("======================================================================")
    print("          STARTING PPO FOUNDATIONAL COMPONENTS VALIDATION             ")
    print("======================================================================")

    # Automatically set up active hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device for Tests: {device}")

    # Initialize Networks and buffers
    actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    critic = CriticNetwork(state_dim=OBSERVATION_DIM).to(device)
    buffer = PPORolloutBuffer()

    # -------------------------------------------------------------------------
    # 1 & 2. Actor accepts a Phase 2 observation of shape (50,) and returns 70 logits.
    # -------------------------------------------------------------------------
    print(f"Validation 1 & 2: Testing Actor with single observation shape ({OBSERVATION_DIM},)...")
    single_obs = torch.randn(OBSERVATION_DIM, device=device)
    logits_single = actor(single_obs)
    assert logits_single.shape == (ACTION_DIM,), f"Expected logits shape ({ACTION_DIM},), got {logits_single.shape}"
    print("-> PASSED: Actor forward pass on single state matches expectations.")

    # -------------------------------------------------------------------------
    # 3. Actor accepts a Phase 2 batch such as (32, 50).
    # -------------------------------------------------------------------------
    print(f"Validation 3: Testing Actor with batch input (32, {OBSERVATION_DIM})...")
    batch_obs = torch.randn(32, OBSERVATION_DIM, device=device)
    logits_batch = actor(batch_obs)
    assert logits_batch.shape == (32, ACTION_DIM), f"Expected logits shape (32, {ACTION_DIM}), got {logits_batch.shape}"
    print("-> PASSED: Actor batch forward pass is successful.")

    # -------------------------------------------------------------------------
    # 4. Critic returns one scalar value per observation
    # -------------------------------------------------------------------------
    print("Validation 4: Testing Critic value function estimations...")
    val_single = critic(single_obs)
    assert val_single.shape == (1,), f"Expected single scalar shape (1,), got {val_single.shape}"
    
    val_batch = critic(batch_obs)
    assert val_batch.shape == (32, 1), f"Expected batch state-values shape (32, 1), got {val_batch.shape}"
    print("-> PASSED: Critic correctly computes V(s) scalar shapes for both single and batched inputs.")

    # -------------------------------------------------------------------------
    # 5 & 6. Masked policy samples only legal actions / Illegal actions have zero probability
    # -------------------------------------------------------------------------
    print("Validation 5 & 6: Testing Masked Categorical Policy action masking rules...")
    # Mock logits where illegal actions have extremely high values before masking is applied
    mock_logits = torch.zeros(ACTION_DIM, device=device)
    mock_logits[0] = 5.0
    mock_logits[1] = 999.0
    mock_logits[2] = 3.0
    mock_logits[3] = 999.0
    # Legal-action mask (Attack 1 and Potion 3 are illegal. End Turn 0, Energy 2, and others are legal)
    mock_mask = torch.ones(ACTION_DIM, device=device)
    mock_mask[1] = 0.0
    mock_mask[3] = 0.0
    
    masked_policy = MaskedCategorical(mock_logits, mock_mask)
    probs = masked_policy.probs
    
    # Assert that probability of masked actions is effectively 0.0
    assert probs[1].item() == 0.0, f"Expected probability of action 1 to be 0.0, got {probs[1].item()}"
    assert probs[3].item() == 0.0, f"Expected probability of action 3 to be 0.0, got {probs[3].item()}"
    
    # Assert legal actions have proper probabilities
    assert probs[0].item() > 0.0, "Legal action 0 should have non-zero probability"
    assert probs[2].item() > 0.0, "Legal action 2 should have non-zero probability"
    
    # Run 1000 sampling trials and verify that no illegal actions are ever generated
    for _ in range(1000):
        action_sample = masked_policy.sample()
        sampled_idx = action_sample.item()
        assert mock_mask[sampled_idx].item() == 1.0, f"Sampled illegal masked action index: {sampled_idx}!"
        
    print("-> PASSED: Action masking verified. Illegal moves have 0.0 probability and are never sampled.")

    # -------------------------------------------------------------------------
    # 6b. A forced-promotion mask with only action 68 legal remains finite.
    # -------------------------------------------------------------------------
    print("Validation 6b: Testing a single-legal-action forced-promotion mask...")
    forced_logits = torch.zeros(ACTION_DIM, device=device)
    forced_logits[0] = 9999.0  # Illegal preference must not leak through the mask.
    forced_mask = torch.zeros(ACTION_DIM, device=device)
    forced_mask[68] = 1.0
    forced_policy = MaskedCategorical(forced_logits, forced_mask)
    assert forced_policy.probs[68].item() == 1.0
    assert forced_policy.probs[0].item() == 0.0
    forced_action = forced_policy.sample()
    assert forced_action.item() == 68
    assert torch.isfinite(forced_policy.log_prob(forced_action))
    assert torch.isfinite(forced_policy.entropy())
    print("-> PASSED: Forced promotion only samples action 68 without numerical instability.")

    # -------------------------------------------------------------------------
    # 7. log_prob() works for sampled legal actions
    # -------------------------------------------------------------------------
    print("Validation 7: Verifying action log probability calculation...")
    sampled_action = masked_policy.sample()
    log_prob_val = masked_policy.log_prob(sampled_action)
    assert not torch.isnan(log_prob_val) and not torch.isinf(log_prob_val), f"Log prob is invalid: {log_prob_val}"
    print(f"-> PASSED: Log prob calculation works correctly. Log-probability: {log_prob_val.item():.4f}")

    # -------------------------------------------------------------------------
    # 8. entropy() is finite
    # -------------------------------------------------------------------------
    print("Validation 8: Verifying Shannon entropy of policy...")
    ent_val = masked_policy.entropy()
    assert not torch.isnan(ent_val) and not torch.isinf(ent_val), f"Entropy is invalid: {ent_val}"
    assert ent_val.item() > 0.0, "Entropy of non-deterministic distribution should be positive"
    print(f"-> PASSED: Shannon policy entropy is finite: {ent_val.item():.4f}")

    # -------------------------------------------------------------------------
    # 9. RolloutBuffer stores transitions correctly
    # -------------------------------------------------------------------------
    print("Validation 9: Testing PPORolloutBuffer storage...")
    dummy_state = np.ones(OBSERVATION_DIM, dtype=np.float32)
    dummy_mask = np.ones(ACTION_DIM, dtype=np.float32)
    dummy_mask[1] = 0.0  # Make action 1 illegal
    
    buffer.add(
        state=dummy_state,
        action=0,
        reward=1.0,
        done=False,
        value=0.5,
        log_prob=-0.2,
        action_mask=dummy_mask
    )
    assert buffer.length() == 1, f"Expected buffer length 1, got {buffer.length()}"
    print("-> PASSED: Transitions successfully stored inside PPORolloutBuffer.")

    # -------------------------------------------------------------------------
    # 10. RolloutBuffer returns correct tensor shapes
    # -------------------------------------------------------------------------
    print("Validation 10: Verifying buffer tensor conversion and shapes...")
    # Add a second experience transition to test multiple rows
    buffer.add(
        state=dummy_state * 2,
        action=2,
        reward=-0.5,
        done=True,
        value=-0.1,
        log_prob=-0.9,
        action_mask=dummy_mask
    )
    assert len(buffer) == 2
    
    # Retrieve Tensors
    states, actions, rewards, dones, values, log_probs, masks = buffer.get_tensors(device)
    
    assert states.shape == (2, OBSERVATION_DIM), f"Expected states shape (2, {OBSERVATION_DIM}), got {states.shape}"
    assert actions.shape == (2,), f"Expected actions shape (2,), got {actions.shape}"
    assert rewards.shape == (2,), f"Expected rewards shape (2,), got {rewards.shape}"
    assert dones.shape == (2,), f"Expected dones shape (2,), got {dones.shape}"
    assert values.shape == (2,), f"Expected values shape (2,), got {values.shape}"
    assert log_probs.shape == (2,), f"Expected log_probs shape (2,), got {log_probs.shape}"
    assert masks.shape == (2, ACTION_DIM), f"Expected masks shape (2, {ACTION_DIM}), got {masks.shape}"
    
    # Ensure tensors are placed on the correct device
    assert states.device.type == device.type, "Tensors not placed on the active device!"
    print("-> PASSED: Experience lists successfully converted to correct tensor shapes on active device.")

    # -------------------------------------------------------------------------
    # 11. Buffer clear() works
    # -------------------------------------------------------------------------
    print("Validation 11: Testing RolloutBuffer flush clear()...")
    buffer.clear()
    assert buffer.length() == 0, f"Expected cleared buffer length 0, got {buffer.length()}"
    print("-> PASSED: PPORolloutBuffer clears cleanly.")

    # -------------------------------------------------------------------------
    # 12 & 13. CPU and CUDA Execution safety
    # -------------------------------------------------------------------------
    print("Validation 12 & 13: Verifying CPU/CUDA hardware safety...")
    # Explicitly verify CPU device execution is functional
    cpu_device = torch.device("cpu")
    actor_cpu = ActorNetwork().to(cpu_device)
    critic_cpu = CriticNetwork().to(cpu_device)
    out_cpu = actor_cpu(torch.randn(1, OBSERVATION_DIM, device=cpu_device))
    assert out_cpu.device.type == "cpu", "CPU forward pass failed to output on CPU."
    print("-> PASSED: CPU execution behaves identically and safely.")
    
    if torch.cuda.is_available():
        print("CUDA GPU is available! Verification on CUDA:")
        cuda_device = torch.device("cuda")
        actor_cuda = actor.to(cuda_device)
        out_cuda = actor_cuda(torch.randn(1, OBSERVATION_DIM, device=cuda_device))
        assert out_cuda.device.type == "cuda", "CUDA execution failed to output on GPU."
        print("-> PASSED: CUDA GPU execution is verified and stable.")
    else:
        print("-> INFO: CUDA GPU not available on this environment (Skipping CUDA test).")

    print("\n======================================================================")
    print("          ALL PPO COMPONENT VALIDATION TESTS PASSED CLEANLY!          ")
    print("======================================================================")


if __name__ == "__main__":
    test_ppo_components()
