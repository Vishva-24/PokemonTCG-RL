import torch
import numpy as np
import random
from pokemon_rl.agents.dqn import DQNetwork, ReplayBuffer, select_action
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def test_dqn_components():
    print("======================================================================")
    print("          STARTING DQN COMPONENTS VALIDATION SUITE                    ")
    print("======================================================================")

    # 1. DQN Network Initialization & Single Observation Shape test
    print("Validation 1: Verifying DQN accepts Phase 2 observations of shape (50,)...")
    device = torch.device("cpu")
    model = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    
    single_obs = np.random.uniform(0.0, 100.0, size=(OBSERVATION_DIM,)).astype(np.float32)
    single_obs_t = torch.tensor(single_obs, dtype=torch.float32).to(device)
    
    # 2. Network returns exactly 70 Q-values
    print("Validation 2: Verifying network returns exactly 70 Q-values...")
    model.eval()
    with torch.no_grad():
        q_values = model(single_obs_t)
        
    assert q_values.shape == (ACTION_DIM,), f"Expected output shape ({ACTION_DIM},), got {q_values.shape}"
    print("-> PASSED: DQNetwork output shape is correct for single observation.")

    # 3. Batch of observations also works
    print("Validation 3: Verifying batched input handling...")
    batch_size = 32
    batch_obs = np.random.uniform(0.0, 100.0, size=(batch_size, OBSERVATION_DIM)).astype(np.float32)
    batch_obs_t = torch.tensor(batch_obs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        batch_q_values = model(batch_obs_t)
        
    assert batch_q_values.shape == (batch_size, ACTION_DIM), \
        f"Expected output shape ({batch_size}, {ACTION_DIM}), got {batch_q_values.shape}"
    print("-> PASSED: DQNetwork successfully processed batched Phase 2 inputs.")

    # 4. Replay Buffer can store transitions
    print("Validation 4: Verifying experience replay buffer storage...")
    buffer = ReplayBuffer(capacity=100)
    assert len(buffer) == 0
    
    state = np.random.uniform(0.0, 100.0, size=(OBSERVATION_DIM,)).astype(np.float32)
    next_state = np.random.uniform(0.0, 100.0, size=(OBSERVATION_DIM,)).astype(np.float32)
    
    # Add a transition
    buffer.add(state, action=2, reward=1.0, next_state=next_state, done=False)
    assert len(buffer) == 1
    print("-> PASSED: ReplayBuffer stores experiences successfully.")

    # 5. Replay Buffer can sample a minibatch with correct shapes
    print("Validation 5: Verifying ReplayBuffer minibatch sampling and shapes...")
    # Fill the buffer up
    for _ in range(50):
        s = np.random.uniform(0.0, 100.0, size=(OBSERVATION_DIM,)).astype(np.float32)
        ns = np.random.uniform(0.0, 100.0, size=(OBSERVATION_DIM,)).astype(np.float32)
        buffer.add(s, action=random.randint(0, ACTION_DIM - 1), reward=float(random.choice([-1.0, 0.0, 1.0])), next_state=ns, done=random.choice([True, False]))
        
    assert len(buffer) == 51
    
    # Sample batch
    states_t, actions_t, rewards_t, next_states_t, dones_t = buffer.sample(batch_size=16)
    
    assert states_t.shape == (16, OBSERVATION_DIM), f"Expected Phase 2 states shape (16, {OBSERVATION_DIM}), got {states_t.shape}"
    assert actions_t.shape == (16, 1), f"Expected actions shape (16, 1), got {actions_t.shape}"
    assert rewards_t.shape == (16, 1), f"Expected rewards shape (16, 1), got {rewards_t.shape}"
    assert next_states_t.shape == (16, OBSERVATION_DIM), f"Expected Phase 2 next states shape (16, {OBSERVATION_DIM}), got {next_states_t.shape}"
    assert dones_t.shape == (16, 1), f"Expected dones shape (16, 1), got {dones_t.shape}"
    print("-> PASSED: Buffer successfully sampled minibatch of size 16 with correct PyTorch Tensor shapes.")

    # 6. Epsilon-greedy selection never selects an illegal action when an action mask is supplied
    print("Validation 6: Testing that epsilon-greedy exploration never selects an illegal action...")
    action_mask = np.zeros(ACTION_DIM, dtype=np.float32)
    # Mark a normal action and the two forced-promotion actions as legal.
    legal_indices = [4, 7, 68, 69]
    for idx in legal_indices:
        action_mask[idx] = 1.0
        
    # Run 1000 simulated random steps with epsilon=1.0 (pure exploration)
    for _ in range(1000):
        selected = select_action(state, model, action_mask, epsilon=1.0, device=device)
        assert selected in legal_indices, f"Epsilon random selected illegal action {selected}! Legal are {legal_indices}"
    print("-> PASSED: Epsilon exploration strictly respects action masking across 1000 trials.")

    # 7. Greedy selection selects the highest-Q legal action
    print("Validation 7: Verifying greedy selection picks the highest-Q legal action...")
    # Determine which of the legal_indices has the highest Q value
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        raw_q = model(state_t).squeeze(0).numpy()
    
    expected_best = legal_indices[np.argmax(raw_q[legal_indices])]
    
    selected_best = select_action(state, model, action_mask, epsilon=0.0, device=device)
    assert selected_best == expected_best, \
        f"Greedy action selected index {selected_best}, expected {expected_best}. Q-values: {raw_q}"
    print(f"-> PASSED: Greedy selection successfully chose best action {selected_best} (Q-value: {raw_q[selected_best]:.4f}).")

    # 8. CPU compatibility verification
    print("Validation 8: Verifying CPU execution...")
    assert device.type == "cpu", f"Test is running on {device}, expected CPU."
    print("-> PASSED: All operations successfully verified on CPU.")

    print("\n======================================================================")
    print("          ALL DQN COMPONENTS VALIDATION TESTS PASSED CLEANLY!         ")
    print("======================================================================")

if __name__ == "__main__":
    test_dqn_components()
