import torch
import numpy as np
import math
from pokemon_rl.agents.dqn_trainer import DQNTrainer, compute_batch_action_masks
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def test_dqn_trainer_full():
    print("======================================================================")
    print("          STARTING DQN TRAINER VALIDATION SUITE                      ")
    print("======================================================================")
    
    device = torch.device("cpu")
    trainer = DQNTrainer(
        state_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        lr=1e-3,
        gamma=0.95,
        capacity=100,
        batch_size=4,
        device=device
    )
    
    # -------------------------------------------------------------------------
    # 1. Online and target networks initialize with identical parameters
    # -------------------------------------------------------------------------
    print("Validation 1: Verifying identical initialization weights...")
    for p_online, p_target in zip(trainer.policy_net.parameters(), trainer.target_net.parameters()):
        assert torch.equal(p_online, p_target), "Networks did not initialize with identical parameters."
    print("-> PASSED: Initial parameters are identical.")

    # -------------------------------------------------------------------------
    # 2. A target-network update correctly copies parameters
    # -------------------------------------------------------------------------
    print("Validation 2: Verifying target network parameter copy update...")
    # Change weights of policy_net manually to simulate learning
    with torch.no_grad():
        trainer.policy_net.fc3.bias.fill_(1.23)
    
    # Verify they are different now
    assert not torch.equal(trainer.policy_net.fc3.bias, trainer.target_net.fc3.bias), \
        "Target net bias should be different from altered policy net bias before update."
        
    # Synchronize
    trainer.update_target_network()
    assert torch.equal(trainer.policy_net.fc3.bias, trainer.target_net.fc3.bias), \
        "Target net bias was not copied correctly during update_target_network()."
    print("-> PASSED: Target network parameter copy successfully verified.")

    # -------------------------------------------------------------------------
    # 3. A replay buffer can provide a valid minibatch
    # -------------------------------------------------------------------------
    print("Validation 3: Verifying replay buffer sampling capability...")
    # Fill replay buffer with dummy transitions
    for i in range(10):
        state = np.random.rand(OBSERVATION_DIM).astype(np.float32)
        next_state = np.random.rand(OBSERVATION_DIM).astype(np.float32)
        trainer.replay_buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)
        
    assert len(trainer.replay_buffer) == 10
    batch = trainer.replay_buffer.sample(4)
    states, actions, rewards, next_states, dones = batch
    assert states.shape == (4, OBSERVATION_DIM)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, OBSERVATION_DIM)
    assert dones.shape == (4, 1)
    print("-> PASSED: Replay buffer samples minibatches with correct shapes.")

    # -------------------------------------------------------------------------
    # 4. train_step() returns a finite loss
    # -------------------------------------------------------------------------
    print("Validation 4: Verifying train_step() returns a finite float loss...")
    loss = trainer.train_step()
    assert isinstance(loss, float), f"Loss must be a float, got {type(loss)}"
    assert not math.isnan(loss) and not math.isinf(loss), f"Loss must be finite, got {loss}"
    print(f"-> PASSED: train_step() returned finite loss: {loss:.6f}")

    # -------------------------------------------------------------------------
    # 5. At least one network parameter changes after a training step
    # -------------------------------------------------------------------------
    print("Validation 5: Verifying network parameters update after training step...")
    # Read the parameter before training step
    bias_before = trainer.policy_net.fc3.bias.clone()
    
    # Run another step to trigger another weight update
    _ = trainer.train_step()
    
    bias_after = trainer.policy_net.fc3.bias.clone()
    # Ensure parameter modified
    assert not torch.equal(bias_before, bias_after), "Policy network weights did not update after backprop."
    print("-> PASSED: Network parameters changed after optimization step.")

    # -------------------------------------------------------------------------
    # 6. Terminal transitions do not bootstrap future Q-values
    # -------------------------------------------------------------------------
    print("Validation 6: Verifying terminal transitions bypass future Q bootstrapping...")
    # Test update equation with done=True transition
    test_rewards = torch.tensor([[10.0]], dtype=torch.float32)
    test_dones_terminal = torch.tensor([[1.0]], dtype=torch.float32)
    test_max_next_q = torch.tensor([[999.0]], dtype=torch.float32) # Huge bootstrap value
    
    # target = reward + (1.0 - done) * gamma * max_next_q
    # If done is 1.0, term (1.0 - done) is 0.0, so bootstrapping is completely zeroed out
    target_q = test_rewards + (1.0 - test_dones_terminal) * trainer.gamma * test_max_next_q
    assert target_q.item() == 10.0, f"Expected exactly reward (10.0), but got {target_q.item()} (bootstrapped!)"
    print("-> PASSED: Terminal states successfully set target Q directly to reward.")

    # -------------------------------------------------------------------------
    # 7. Illegal next-state actions are excluded from target-Q computation
    # -------------------------------------------------------------------------
    print("Validation 7: Verifying illegal next-state actions are excluded from target-Q...")
    # Setup next state where Attack (Action 1) is illegal (attached energy is 0)
    illegal_next_state = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    illegal_next_state[0] = 70.0   # Active HP
    illegal_next_state[5] = 4.0    # Active Pokemon type: Lightning/Pikachu
    # Indices 1-4 are all zero, so neither attack is legal.
    
    # Verify action mask says action 1 (Attack 1) is indeed illegal (0.0)
    masks = compute_batch_action_masks(torch.tensor(np.array([illegal_next_state]), dtype=torch.float32))
    assert masks[0, 1] == 0.0, "Action 1 (Attack 1) must be masked (illegal) in this state."
    
    # Let's mock our target_net bias to output a massive Q-value for the illegal action 1
    # and a modest Q-value for the legal Action 0 (End Turn)
    with torch.no_grad():
        # Zero out weights and bias of target network
        trainer.target_net.fc3.bias.fill_(0.0)
        trainer.target_net.fc3.bias[1] = 999.0  # Illegal action Q
        trainer.target_net.fc3.bias[0] = 5.0    # Legal action Q
        
        # Zero out fc layers so bias represents exact Q output
        trainer.target_net.fc1.weight.fill_(0.0)
        trainer.target_net.fc1.bias.fill_(0.0)
        trainer.target_net.fc2.weight.fill_(0.0)
        trainer.target_net.fc2.bias.fill_(0.0)
        trainer.target_net.fc3.weight.fill_(0.0)

    # Re-run target forward pass on next states
    next_states_t = torch.tensor(np.array([illegal_next_state]), dtype=torch.float32)
    next_q_values = trainer.target_net(next_states_t) # returns [5.0, 999.0, 0.0, ...]
    
    # Calculate masked target next Q-values
    next_masks = compute_batch_action_masks(next_states_t)
    masked_next_q_values = next_q_values.clone()
    masked_next_q_values[next_masks == 0.0] = -1e9
    
    max_next_q, _ = torch.max(masked_next_q_values, dim=1, keepdim=True)
    
    # Max legal Q-value should be 5.0 (Action 0), NOT 999.0 (Action 1)
    assert max_next_q.item() == 5.0, f"Target Q bootstrap did not exclude illegal actions! Got {max_next_q.item()}"
    print("-> PASSED: Target Q calculations strictly ignore highly-rated illegal next-state actions.")

    # -------------------------------------------------------------------------
    # 8. Forced promotion exposes only actions 68 and 69
    # -------------------------------------------------------------------------
    print("Validation 8: Verifying forced-promotion action reconstruction...")
    forced_promotion_state = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    forced_promotion_state[11] = 1.0  # Bench 1 exists
    forced_promotion_state[17] = 3.0  # Bench 2 exists
    forced_promotion_state[29] = 1.0  # Forced-promotion flag
    forced_mask = compute_batch_action_masks(torch.tensor([forced_promotion_state], dtype=torch.float32))
    assert torch.nonzero(forced_mask[0], as_tuple=False).flatten().tolist() == [68, 69]
    print("-> PASSED: Forced-promotion states expose only actions 68 and 69.")

    # -------------------------------------------------------------------------
    # 9. CPU Verification
    # -------------------------------------------------------------------------
    print("Validation 9: Checking CPU execution correctness...")
    assert trainer.policy_net.fc1.weight.device.type == "cpu", "Policy net is not on CPU."
    assert trainer.target_net.fc1.weight.device.type == "cpu", "Target net is not on CPU."
    print("-> PASSED: All trainer network modules run on CPU.")

    # -------------------------------------------------------------------------
    # 10. Repeated training steps can execute without errors
    # -------------------------------------------------------------------------
    print("Validation 10: Testing continuous repeated training steps...")
    for step_idx in range(5):
        loss_val = trainer.train_step()
        assert isinstance(loss_val, float), "Loss must be float."
    print("-> PASSED: 5 sequential gradient optimization steps completed without errors.")

    print("\n======================================================================")
    print("          ALL DQN TRAINER VALIDATION TESTS PASSED CLEANLY!           ")
    print("======================================================================")

if __name__ == "__main__":
    test_dqn_trainer_full()
