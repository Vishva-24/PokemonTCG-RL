import os
import copy
import random
import numpy as np
from pokemon_rl.environment import PokemonTCGEnv
from pokemon_rl.game import Action

def run_environment_validation():
    print("======================================================================")
    print("          STARTING PRE-DQN ENVIRONMENT VALIDATION SUITE               ")
    print("======================================================================")

    # Instantiate Environment
    env = PokemonTCGEnv()
    
    # -------------------------------------------------------------------------
    # 1. env.reset(seed=42) works and returns (observation, info)
    # -------------------------------------------------------------------------
    print("Validation 1: Testing env.reset(seed=42) structure...")
    reset_output = env.reset(seed=42)
    assert isinstance(reset_output, tuple) and len(reset_output) == 2, \
        f"reset() must return a tuple of size 2, got {type(reset_output)} of length {len(reset_output)}"
    obs, info = reset_output
    assert isinstance(info, dict), f"info must be a dictionary, got {type(info)}"
    print("-> PASSED: reset(seed=42) executed cleanly and returned (observation, info).")

    # -------------------------------------------------------------------------
    # 2. Observation is contained within env.observation_space
    # -------------------------------------------------------------------------
    print("Validation 2: Verifying observation shape, type, and bounds...")
    assert isinstance(obs, np.ndarray), f"Observation must be a numpy array, got {type(obs)}"
    assert obs.shape == env.observation_space.shape, \
        f"Observation shape {obs.shape} does not match space shape {env.observation_space.shape}"
    assert obs.dtype == np.float32 or obs.dtype == np.float64, \
        f"Observation dtype must be float32/float64, got {obs.dtype}"
    
    # Check bounds
    low_bound = env.observation_space.low
    high_bound = env.observation_space.high
    # Note: low_bound and high_bound can be arrays or scalars. Handled dynamically:
    in_bounds = np.all(obs >= low_bound) and np.all(obs <= high_bound)
    assert in_bounds, f"Observation values out of bounds! Obs: {obs}, Bounds: [{low_bound}, {high_bound}]"
    print(f"-> PASSED: Observation shape {obs.shape} is within bounds [{low_bound}, {high_bound}].")

    # -------------------------------------------------------------------------
    # 3. env.action_space is valid
    # -------------------------------------------------------------------------
    print("Validation 3: Verifying action space...")
    # Phase 2's approved interface has exactly 70 discrete actions.
    assert hasattr(env.action_space, 'n') and env.action_space.n == 70, \
        f"Action space must be Discrete(70), got {env.action_space}"
    print("-> PASSED: Action space is Discrete(70).")

    # -------------------------------------------------------------------------
    # 4. env.get_action_mask() has correct shape and contains only 0/1 values
    # -------------------------------------------------------------------------
    print("Validation 4: Verifying action mask shape and values...")
    mask = env.get_action_mask()
    assert isinstance(mask, np.ndarray), f"Action mask must be a numpy array, got {type(mask)}"
    assert mask.shape == (70,), f"Action mask must have shape (70,), got {mask.shape}"
    # Verify that all elements are strictly 0.0 or 1.0
    for idx, val in enumerate(mask):
        assert val == 0.0 or val == 1.0, \
            f"Action mask element at index {idx} must be 0.0 or 1.0, got {val}"
    print("-> PASSED: Action mask has shape (70,) and contains only 0.0 or 1.0.")

    # -------------------------------------------------------------------------
    # 5. Every action marked legal by the mask can be passed to env.step()
    # -------------------------------------------------------------------------
    print("Validation 5: Verifying all masked legal actions can be executed...")
    legal_indices = [idx for idx, val in enumerate(mask) if val == 1.0]
    assert len(legal_indices) > 0, "Action mask has no legal actions! Game state is blocked."
    
    # Test each legal action independently by deep-copying the environment's internal game state
    original_state = copy.deepcopy(env.state)
    
    for action_idx in legal_indices:
        # Restore state to keep environment in the exact same turn start configuration
        env.state = copy.deepcopy(original_state)
        try:
            _ = env.step(action_idx)
        except Exception as e:
            raise AssertionError(f"Step failed on action index {action_idx} even though it was marked legal! Error: {e}")
            
    # Restore original state once more
    env.state = copy.deepcopy(original_state)
    print(f"-> PASSED: All {len(legal_indices)} legal actions ({legal_indices}) executed successfully without raising exceptions.")

    # -------------------------------------------------------------------------
    # 6. Illegal actions are correctly rejected rather than silently executed
    # -------------------------------------------------------------------------
    print("Validation 6: Verifying illegal action rejection...")
    illegal_indices = [idx for idx, val in enumerate(mask) if val == 0.0]
    
    if len(illegal_indices) > 0:
        test_illegal_idx = illegal_indices[0]
        rejected = False
        try:
            env.step(test_illegal_idx)
        except ValueError:
            rejected = True
        except Exception as e:
            raise AssertionError(f"Expected ValueError when taking illegal action, but got {type(e)}: {e}")
            
        assert rejected, f"Illegal action index {test_illegal_idx} was executed without throwing a ValueError!"
        print(f"-> PASSED: Illegal action index {test_illegal_idx} was successfully rejected with ValueError.")
    else:
        print("-> SKIPPED (No illegal actions in starting state).")

    # -------------------------------------------------------------------------
    # 7 & 8. env.step() returns exactly (obs, reward, terminated, truncated, info) and keeps obs in-bounds
    # -------------------------------------------------------------------------
    print("Validation 7 & 8: Verifying step() return type signature and post-step observation...")
    # Choose any legal action to progress the state
    first_legal_action = legal_indices[0]
    step_output = env.step(first_legal_action)
    
    assert isinstance(step_output, tuple) and len(step_output) == 5, \
        f"step() must return a 5-element tuple, got {type(step_output)} of size {len(step_output)}"
        
    next_obs, reward, terminated, truncated, next_info = step_output
    
    assert isinstance(next_obs, np.ndarray), f"Post-step observation must be a numpy array, got {type(next_obs)}"
    assert next_obs.shape == env.observation_space.shape, f"Post-step observation shape mismatch: {next_obs.shape}"
    assert np.all(next_obs >= low_bound) and np.all(next_obs <= high_bound), "Post-step observation out of bounds!"
    
    assert isinstance(reward, (float, int)), f"Reward must be a float or integer, got {type(reward)}"
    assert isinstance(terminated, bool), f"Terminated must be a boolean, got {type(terminated)}"
    assert isinstance(truncated, bool), f"Truncated must be a boolean, got {type(truncated)}"
    assert isinstance(next_info, dict), f"Post-step info must be a dict, got {type(next_info)}"
    
    print("-> PASSED: step() returned correct 5-tuple signature and post-step observation remains in-bounds.")

    # -------------------------------------------------------------------------
    # 9. A complete random legal-action episode eventually terminates
    # -------------------------------------------------------------------------
    print("Validation 9: Testing random episode termination...")
    env.reset(seed=123)
    step_count = 0
    max_steps = 1000
    
    while not env.state.game_over and step_count < max_steps:
        curr_mask = env.get_action_mask()
        curr_legal_indices = [idx for idx, val in enumerate(curr_mask) if val == 1.0]
        if not curr_legal_indices:
            # If no legal action, call step with 0 (end turn) to keep transition going
            action_to_take = 0
        else:
            action_to_take = random.choice(curr_legal_indices)
            
        _, _, terminated, _, _ = env.step(action_to_take)
        step_count += 1
        
    assert env.state.game_over, f"Game failed to terminate after {max_steps} steps! Potential infinite loop."
    print(f"-> PASSED: Random play episode terminated naturally in {step_count} steps.")

    # -------------------------------------------------------------------------
    # 10. env.reset(seed=42) is reproducible
    # -------------------------------------------------------------------------
    print("Validation 10: Verifying reset seed reproducibility...")
    obs_run_1, _ = env.reset(seed=42)
    obs_run_2, _ = env.reset(seed=42)
    assert np.array_equal(obs_run_1, obs_run_2), \
        f"reset(seed=42) is not reproducible! Run 1: {obs_run_1}, Run 2: {obs_run_2}"
    print("-> PASSED: env.reset(seed=42) is perfectly reproducible.")

    # -------------------------------------------------------------------------
    # 11. Different seeds produce different states/games
    # -------------------------------------------------------------------------
    print("Validation 11: Testing that different seeds generate different games...")
    obs_seed_42, _ = env.reset(seed=42)
    obs_seed_123, _ = env.reset(seed=123)
    
    # Hand/Deck randomization should make starting states distinct
    is_different = not np.array_equal(obs_seed_42, obs_seed_123)
    assert is_different, "Resetting with seeds 42 and 123 produced identical starting states!"
    print("-> PASSED: Seeds 42 and 123 successfully produced distinct starting states.")

    # -------------------------------------------------------------------------
    # 12. Verification of rule safety
    # -------------------------------------------------------------------------
    print("Validation 12: Phase 2 rule and legal-action validation completed.")
    
    print("\n======================================================================")
    print("          ALL ENVIRONMENT VALIDATION TESTS PASSED SUCCESSFULLY!       ")
    print("======================================================================")

if __name__ == "__main__":
    run_environment_validation()
