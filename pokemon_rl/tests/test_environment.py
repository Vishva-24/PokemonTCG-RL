import numpy as np
from pokemon_rl.environment import PokemonTCGEnv

def test_gym_environment():
    print("--- TESTING GYMNASIUM WRAPPER: INITIALIZATION ---")
    
    # 1. Create environment
    env = PokemonTCGEnv()
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # 2. Reset environment
    obs, info = env.reset(seed=42)
    print("\n--- ENVIRONMENT RESET ---")
    print(f"Initial Observation Vector: {obs}")
    print(f"Observation Shape: {obs.shape}")
    print(f"Initial Info: {info}")
    
    # 3. Print the action mask
    mask = env.get_action_mask()
    print(f"\nInitial Action Mask (size 70): {mask}")
    
    # Verify mapping correctness
    assert obs.shape == (50,)
    assert mask.shape == (70,)
    assert mask[0] == 1.0  # End Turn should be legal
    
    # 4. Execute a couple of sample steps manually
    print("\n--- EXECUTING SAMPLE MANUAL STEPS ---")
    
    # Sample a legal action from the mask
    legal_indices = np.where(mask == 1.0)[0]
    print(f"Legal Action Indices in this state: {legal_indices}")
    
    # Let's attach an energy card if legal
    energy_actions = [idx for idx in legal_indices if 20 <= idx <= 27]
    if energy_actions:
        chosen_idx = energy_actions[0]
        print(f"Agent chose to play Attach Energy (Index: {chosen_idx})")
    else:
        chosen_idx = legal_indices[0]
        print(f"Agent chose action (Index: {chosen_idx})")
        
    obs, reward, terminated, truncated, info = env.step(chosen_idx)
    print(f"After Step 1 -> Reward: {reward}, Terminated: {terminated}, Turn Count: {info['turn_count']}")
    print(f"New Observation: {obs}")
    print(f"Updated Action Mask: {env.get_action_mask()}")

    # 5. Run a complete automated game with random actions
    print("\n--- SIMULATING A COMPLETE RANDOM-AGENT GAME ---")
    obs, info = env.reset(seed=100)
    terminated = False
    step_count = 0
    
    while not terminated and step_count < 200:
        mask = env.get_action_mask()
        legal_indices = np.where(mask == 1.0)[0]
        
        if len(legal_indices) == 0:
            print("No legal actions! Forcing End Turn (Index 0).")
            action_idx = 0
        else:
            action_idx = np.random.choice(legal_indices)
            
        obs, reward, terminated, truncated, info = env.step(action_idx)
        step_count += 1
        
    print(f"\nGame Terminated naturally after {step_count} agent steps.")
    print(f"Final Info: {info}")
    print(f"Final Reward received: {reward}")
    
    # Assertions
    assert terminated is True
    assert reward in [1.0, -1.0]
    assert info["winner"] in ["Red", "Blue"]
    print("\nGymnasium environment test completed and passed successfully!")

if __name__ == "__main__":
    test_gym_environment()
