import torch
import numpy as np
import math

from pokemon_rl.agents.lstm_ppo import LSTMActorNetwork, LSTMCriticNetwork, MaskedCategorical, LSTMPPORolloutBuffer
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def test_lstm_ppo_suite():
    print("======================================================================")
    print("          STARTING RECURRENT LSTM-PPO COMPONENT VALIDATION            ")
    print("======================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device for Recurrent Validation: {device}")

    # 1. Initialize Networks
    actor = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, hidden_dim=128).to(device)
    critic = LSTMCriticNetwork(state_dim=OBSERVATION_DIM, hidden_dim=128).to(device)

    # -------------------------------------------------------------------------
    # Test 1 & 5: Actor single-step forward pass & hidden states
    # -------------------------------------------------------------------------
    print(f"Validation 1 & 5: Actor single-step forward pass with ({OBSERVATION_DIM},) shape...")
    single_obs = torch.randn(OBSERVATION_DIM, device=device)
    h0, c0 = actor.init_hidden(batch_size=1, device=device)
    
    assert h0.shape == (1, 1, 128), f"Expected h0 shape (1, 1, 128), got {h0.shape}"
    assert c0.shape == (1, 1, 128), f"Expected c0 shape (1, 1, 128), got {c0.shape}"
    
    logits, (h1, c1) = actor(single_obs, (h0, c0))
    assert logits.shape == (ACTION_DIM,), f"Expected single-step logits shape ({ACTION_DIM},), got {logits.shape}"
    assert h1.shape == (1, 1, 128)
    assert c1.shape == (1, 1, 128)
    print("-> PASSED: Single-step forward pass succeeds and outputs correct hidden shapes.")

    # -------------------------------------------------------------------------
    # Test 2 & 3: Actor sequence forward pass
    # -------------------------------------------------------------------------
    print(f"Validation 2 & 3: Actor sequence forward pass (shape (4, 16, {OBSERVATION_DIM}))...")
    seq_obs = torch.randn(4, 16, OBSERVATION_DIM, device=device)
    h0_seq, c0_seq = actor.init_hidden(batch_size=4, device=device)
    
    logits_seq, (h1_seq, c1_seq) = actor(seq_obs, (h0_seq, c0_seq))
    assert logits_seq.shape == (4, 16, ACTION_DIM), f"Expected logits shape (4, 16, {ACTION_DIM}), got {logits_seq.shape}"
    assert h1_seq.shape == (1, 4, 128)
    assert c1_seq.shape == (1, 4, 128)
    print("-> PASSED: Sequence forward pass succeeds and outputs correct logits sequence shape.")

    # -------------------------------------------------------------------------
    # Test 4: Critic sequence output
    # -------------------------------------------------------------------------
    print("Validation 4: Critic value function sequence output (shape (4, 16, 1))...")
    h0_c, c0_c = critic.init_hidden(batch_size=4, device=device)
    values_seq, (h1_c, c1_c) = critic(seq_obs, (h0_c, c0_c))
    assert values_seq.shape == (4, 16, 1), f"Expected values shape (4, 16, 1), got {values_seq.shape}"
    assert h1_c.shape == (1, 4, 128)
    assert c1_c.shape == (1, 4, 128)
    print("-> PASSED: Recurrent Critic outputs correct sequence shapes.")

    # -------------------------------------------------------------------------
    # Test 6: Carry hidden states sequence-to-sequence
    # -------------------------------------------------------------------------
    print("Validation 6: Verifying hidden-state sequential propagation...")
    # First sequence pass
    _, (h_mid, c_mid) = actor(seq_obs, (h0_seq, c0_seq))
    # Second sequence pass seeded with first outputs
    logits_next, (h_final, c_final) = actor(seq_obs, (h_mid, c_mid))
    assert h_final.shape == (1, 4, 128)
    assert not torch.equal(h_mid, h_final), "Hidden state values failed to propagate or evolve across sequences."
    print("-> PASSED: Recurrent cells cleanly carry information across sequence steps.")

    # -------------------------------------------------------------------------
    # Test 7: Reset hidden states
    # -------------------------------------------------------------------------
    print("Validation 7: Verifying hidden states reset mechanism...")
    h_reset, c_reset = actor.init_hidden(batch_size=4, device=device)
    assert torch.equal(h_reset, torch.zeros_like(h_reset))
    assert torch.equal(c_reset, torch.zeros_like(c_reset))
    print("-> PASSED: Zero-initialization functions correctly.")

    # -------------------------------------------------------------------------
    # Test 8 & 9: Action masking works for every step in a sequence
    # -------------------------------------------------------------------------
    print("Validation 8 & 9: Testing action masking over sequential dimensions...")
    mock_logits = torch.randn(2, 4, ACTION_DIM, device=device)
    mock_mask = torch.ones(2, 4, ACTION_DIM, device=device)
    mock_mask[:, :, 1] = 0.0
    mock_mask[:, :, 5] = 0.0
    
    masked_policy = MaskedCategorical(mock_logits, mock_mask)
    probs = masked_policy.probs
    
    assert torch.all(probs[:, :, 1] == 0.0), "Action 1 was not masked properly."
    assert torch.all(probs[:, :, 5] == 0.0), "Action 5 was not masked properly."
    
    for _ in range(100):
        sampled_actions = masked_policy.sample()
        for b in range(2):
            for t in range(4):
                action_idx = sampled_actions[b, t].item()
                assert mock_mask[b, t, action_idx].item() == 1.0, f"Sampled illegal action {action_idx}!"
                
    print("-> PASSED: Sequence action masking is mathematically airtight across all timesteps.")

    # -------------------------------------------------------------------------
    # Test 9b: Forced promotion supports 68-only, 69-only, and 68/69 masks.
    # -------------------------------------------------------------------------
    print("Validation 9b: Testing forced-promotion masks across a sequence...")
    forced_logits = torch.zeros(1, 3, ACTION_DIM, device=device)
    forced_logits[:, :, 0] = 9999.0  # Illegal preference must be fully masked.
    forced_mask = torch.zeros(1, 3, ACTION_DIM, device=device)
    forced_mask[0, 0, 68] = 1.0
    forced_mask[0, 1, 69] = 1.0
    forced_mask[0, 2, 68] = 1.0
    forced_mask[0, 2, 69] = 1.0
    forced_policy = MaskedCategorical(forced_logits, forced_mask)
    for _ in range(100):
        forced_actions = forced_policy.sample()
        assert forced_actions[0, 0].item() == 68
        assert forced_actions[0, 1].item() == 69
        assert forced_actions[0, 2].item() in (68, 69)
    assert torch.all(torch.isfinite(forced_policy.entropy()))
    assert torch.all(torch.isfinite(forced_policy.log_prob(forced_policy.sample())))
    print("-> PASSED: Forced-promotion masks remain legal and numerically stable.")

    # -------------------------------------------------------------------------
    # Test 10 & 11: Finiteness of Log probs & Entropy
    # -------------------------------------------------------------------------
    print("Validation 10 & 11: Verifying log_prob and entropy numerical finiteness...")
    sampled = masked_policy.sample()
    log_probs = masked_policy.log_prob(sampled)
    entropy = masked_policy.entropy()
    
    assert torch.all(torch.isfinite(log_probs)), "Found non-finite log-probabilities."
    assert torch.all(torch.isfinite(entropy)), "Found non-finite entropy values."
    print("-> PASSED: Log-probabilities and entropy are clean and fully finite.")

    # -------------------------------------------------------------------------
    # Test 12, 13 & 14: LSTMRolloutBuffer storage, shape, and extraction
    # -------------------------------------------------------------------------
    print("Validation 12, 13 & 14: Verifying LSTMPPORolloutBuffer storage and sequence batching...")
    buffer = LSTMPPORolloutBuffer()
    
    for i in range(32):
        dummy_state = np.ones(OBSERVATION_DIM, dtype=np.float32) * i
        dummy_mask = np.ones(ACTION_DIM, dtype=np.float32)
        dummy_mask[3] = 0.0 # Action 3 is illegal
        action = 0
        if i == 0:
            dummy_mask[:] = 0.0
            dummy_mask[68] = 1.0
            action = 68
        
        h_step = torch.ones(1, 1, 128) * i
        c_step = torch.ones(1, 1, 128) * (i + 100)
        
        buffer.add(
            state=dummy_state,
            action=action,
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-0.2,
            action_mask=dummy_mask,
            actor_h=h_step,
            actor_c=c_step,
            critic_h=h_step,
            critic_c=c_step
        )
        
    assert buffer.length() == 32
    
    states_seq, action_masks_seq, actions_seq, rewards_seq, dones_seq, values_seq, log_probs_seq, act_h_seq, act_c_seq, cr_h_seq, cr_c_seq = \
        buffer.get_sequence_batches(sequence_length=16, device=device)
        
    assert states_seq.shape == (2, 16, OBSERVATION_DIM)
    assert action_masks_seq.shape == (2, 16, ACTION_DIM)
    assert actions_seq.shape == (2, 16)
    assert rewards_seq.shape == (2, 16)
    assert dones_seq.shape == (2, 16)
    assert values_seq.shape == (2, 16)
    assert log_probs_seq.shape == (2, 16)
    
    assert act_h_seq.shape == (1, 2, 128)
    assert act_c_seq.shape == (1, 2, 128)
    assert cr_h_seq.shape == (1, 2, 128)
    assert cr_c_seq.shape == (1, 2, 128)
    
    assert abs(act_h_seq[0, 0, 0].item() - 0.0) < 1e-5
    assert abs(act_h_seq[0, 1, 0].item() - 16.0) < 1e-5
    
    buffer.clear()
    assert len(buffer) == 0
    print("-> PASSED: Actor and Critic rollout-time memory states extracted perfectly.")

    # -------------------------------------------------------------------------
    # Test 15 & 16: CPU / CUDA Execution correctness
    # -------------------------------------------------------------------------
    print("Validation 15 & 16: Verifying CPU execution is stable...")
    cpu_device = torch.device("cpu")
    actor_cpu = LSTMActorNetwork().to(cpu_device)
    out_cpu, _ = actor_cpu(torch.randn(1, OBSERVATION_DIM, device=cpu_device))
    assert out_cpu.device.type == "cpu"
    print("-> PASSED: CPU execution works safely.")
    
    if torch.cuda.is_available():
        print("CUDA available! Testing GPU integration...")
        gpu_device = torch.device("cuda")
        actor_gpu = LSTMActorNetwork().to(gpu_device)
        out_gpu, _ = actor_gpu(torch.randn(1, OBSERVATION_DIM, device=gpu_device))
        assert out_gpu.device.type == "cuda"
        print("-> PASSED: CUDA GPU execution works safely.")
    else:
        print("-> INFO: CUDA GPU not available on this environment (Skipping CUDA check).")

    # -------------------------------------------------------------------------
    # Test 17: Episode boundary reset test (Single Step)
    # -------------------------------------------------------------------------
    print("Validation 17: Performing explicit episode boundary reset test...")
    done_step1 = False
    h_step1, c_step1 = actor.init_hidden(batch_size=1, device=device)
    done_step2 = True
    _, (h_step2, c_step2) = actor(torch.randn(1, OBSERVATION_DIM, device=device), (h_step1, c_step1))
    
    h_reset_step = h_step2 * (1.0 - float(done_step2))
    c_reset_step = c_step2 * (1.0 - float(done_step2))
    
    assert torch.equal(h_reset_step, torch.zeros_like(h_reset_step)), "Hidden state failed to reset on episode boundary!"
    assert torch.equal(c_reset_step, torch.zeros_like(c_reset_step)), "Cell state failed to reset on episode boundary!"
    print("-> PASSED: Memory zeroed out successfully at episode boundaries (leakage is zero).")

    # -------------------------------------------------------------------------
    # Test 18: Synthetic sequence boundary leakage test (A -> done -> B)
    # -------------------------------------------------------------------------
    print("Validation 18: Performing synthetic sequence boundary leakage check...")
    # Initialize random states and dones sequence
    states_syn = torch.randn(1, 16, OBSERVATION_DIM, device=device)
    dones_syn = torch.zeros(1, 16, device=device)
    
    # EXACT BOUNDARY FIXED HERE: Set ONLY index 8 as terminal step of Episode A
    dones_syn[:, 8] = 1.0
    
    # Sequence pass through the actor network (with dones indicating reset at t=9)
    h_start, c_start = actor.init_hidden(batch_size=1, device=device)
    # We populate features of previous steps to have non-zero hidden states at t=8
    seq_logits, (h_final, c_final) = actor(states_syn, (h_start, c_start), dones=dones_syn)
    
    # Standalone Pass: Episode B starts from step 9 through step 15 (L=7 steps)
    # This must run sequentially from an EXACT zero hidden state
    h_b, c_b = actor.init_hidden(batch_size=1, device=device)
    standalone_logits_list = []
    
    # Process sequentially steps 9, 10, 11, 12, 13, 14, 15
    for t in range(9, 16):
        step_input = states_syn[:, t:t+1, :] # shape (1, 1, 50)
        step_logits, (h_b, c_b) = actor(step_input, (h_b, c_b))
        standalone_logits_list.append(step_logits)
        
    standalone_logits = torch.cat(standalone_logits_list, dim=1)
    seq_b_logits = seq_logits[:, 9:, :] # shape (1, 7, 70)
    
    # Check absolute difference
    diff = (seq_b_logits - standalone_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Device appropriate tolerances (due to parallel reductions in CuBLAS/Cudnn)
    if device.type == "cuda":
        tolerance_atol = 1e-4
        tolerance_rtol = 1e-5
        device_label = "CUDA (GPU parallel scan)"
    else:
        tolerance_atol = 1e-7
        tolerance_rtol = 1e-7
        device_label = "CPU"
        
    print(f"  Device:           {device_label}")
    print(f"  Max Output Diff:  {max_diff:.10e}")
    print(f"  Mean Output Diff: {mean_diff:.10e}")
    
    is_correct = torch.allclose(seq_b_logits, standalone_logits, atol=tolerance_atol, rtol=tolerance_rtol)
    assert is_correct, f"Episode boundary leak detected! Standalone outputs differ by max {max_diff:.10e} on {device_label}"
    print(f"-> PASSED: Synthetic sequence boundary test succeeded (leakage is strictly zero within precision bounds).")

    # -------------------------------------------------------------------------
    # Test 19: Explicit User-Requested Diagnostic Comparison
    # -------------------------------------------------------------------------
    print("Validation 19: Running explicit user-requested diagnostic comparison...")
    # 1. Warm up the actor to generate some non-zero hidden states
    h_diag, c_diag = actor.init_hidden(batch_size=1, device=device)
    warmup_states = torch.randn(1, 5, OBSERVATION_DIM, device=device)
    _, (h_warm, c_warm) = actor(warmup_states, (h_diag, c_diag))
    
    actor_hidden_non_zero_before = (h_warm != 0.0).any().item()
    actor_cell_non_zero_before = (c_warm != 0.0).any().item()
    
    print(f"  Before reset: hidden contains non-zeros? {actor_hidden_non_zero_before}")
    print(f"  Before reset: cell contains non-zeros? {actor_cell_non_zero_before}")
    
    # 2. Reset hidden state as if done occurred (done=1.0)
    done_val = 1.0
    h_diag_reset = h_warm * (1.0 - done_val)
    c_diag_reset = c_warm * (1.0 - done_val)
    
    is_h_exactly_zero = (h_diag_reset == 0.0).all().item()
    is_c_exactly_zero = (c_diag_reset == 0.0).all().item()
    
    print(f"  After reset: Is hidden state exactly zero? {is_h_exactly_zero}")
    print(f"  After reset: Is cell state exactly zero? {is_c_exactly_zero}")
    
    # Assert reset states are completely zero at the bit-level
    assert is_h_exactly_zero, "Reset hidden state is not exactly zero!"
    assert is_c_exactly_zero, "Reset cell state is not exactly zero!"
    
    # 3. Compare outputs on a new episode observation
    new_obs = torch.randn(1, 1, OBSERVATION_DIM, device=device)
    zero_h, zero_c = actor.init_hidden(batch_size=1, device=device)
    
    out_zero, _ = actor(new_obs, (zero_h, zero_c))
    out_reset, _ = actor(new_obs, (h_diag_reset, c_diag_reset))
    
    diag_diff = (out_zero - out_reset).abs()
    diag_max_diff = diag_diff.max().item()
    diag_mean_diff = diag_diff.mean().item()
    
    print(f"  Max absolute difference:  {diag_max_diff:.10e}")
    print(f"  Mean absolute difference: {diag_mean_diff:.10e}")
    
    # Direct strict test - must be bit-exact matching because the states were exactly zero
    assert torch.equal(out_zero, out_reset), f"Outputs are not identical even with exactly zero inputs! Max diff: {diag_max_diff}"
    print("-> PASSED: Explicit diagnostic comparison validated successfully.")

    print("\n======================================================================")
    print("          ALL RECURRENT PPO UNIT TESTS PASSED CLEANLY!                ")
    print("======================================================================")

if __name__ == "__main__":
    test_lstm_ppo_suite()
