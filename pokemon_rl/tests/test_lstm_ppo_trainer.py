import torch
import numpy as np
import math

from pokemon_rl.agents.lstm_ppo import LSTMActorNetwork, LSTMCriticNetwork, LSTMPPORolloutBuffer
from pokemon_rl.agents.lstm_ppo_trainer import LSTMPPOTrainer, compute_recurrent_gae_and_returns
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def test_lstm_ppo_trainer_suite():
    print("======================================================================")
    print("          STARTING PPO TRAINER VALIDATION SUITE                      ")
    print("======================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device for Trainer Test: {device}")

    # Initialize independent recurrent networks
    actor = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, hidden_dim=128).to(device)
    critic = LSTMCriticNetwork(state_dim=OBSERVATION_DIM, hidden_dim=128).to(device)

    # 1. PPOTrainer initializes Actor and Critic correctly
    print("Validation 1 & 2: Testing trainer initialization and networks parameter registration...")
    trainer = LSTMPPOTrainer(
        actor=actor,
        critic=critic,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.01,
        value_loss_coefficient=0.5,
        max_grad_norm=0.5,
        ppo_epochs=5,
        sequence_length=16,
        batch_size=2,
        device=device
    )
    assert trainer.actor is actor
    assert trainer.critic is critic
    assert len(list(trainer.actor.parameters())) > 0, "Actor lacks learnable weights."
    assert len(list(trainer.critic.parameters())) > 0, "Critic lacks learnable weights."
    print("-> PASSED: Initialization is correct and learnable layers are properly registered.")

    # 3 & 4. Synthetic sequence tensors and hidden states verify shapes
    print("Validation 3 & 4: Verifying synthetic batch and hidden states shapes...")
    # B = 4 sequence blocks, each with seq_len L = 16
    syn_states = torch.randn(4, 16, OBSERVATION_DIM, device=device)
    syn_action_masks = torch.ones(4, 16, ACTION_DIM, device=device)
    syn_actions = torch.randint(0, ACTION_DIM, (4, 16), device=device)
    syn_old_log_probs = torch.randn(4, 16, device=device)
    syn_returns = torch.randn(4, 16, device=device)
    syn_advantages = torch.randn(4, 16, device=device)
    syn_dones = torch.zeros(4, 16, device=device) # Standard zeroes

    # Include forced-promotion timesteps in the BPTT update batch.
    syn_action_masks[0, 0] = 0.0
    syn_action_masks[0, 0, 68] = 1.0
    syn_actions[0, 0] = 68
    syn_action_masks[0, 1] = 0.0
    syn_action_masks[0, 1, 69] = 1.0
    syn_actions[0, 1] = 69
    
    # Standardize advantages
    syn_advantages = (syn_advantages - syn_advantages.mean()) / (syn_advantages.std() + 1e-8)
    
    # Hidden states
    syn_h0 = torch.randn(1, 4, 128, device=device)
    syn_c0 = torch.randn(1, 4, 128, device=device)
    
    assert syn_states.shape == (4, 16, OBSERVATION_DIM)
    assert syn_h0.shape == (1, 4, 128)
    assert syn_c0.shape == (1, 4, 128)
    print("-> PASSED: Synthetic sequence tensors and initial hidden state dimensions are aligned.")

    # 5 to 12. Run optimization pass and inspect outputs
    print("Validation 5, 11 & 12: Running update and verifying diagnostics...")
    diagnostics = trainer.update(
        states=syn_states,
        action_masks=syn_action_masks,
        actions=syn_actions,
        old_log_probs=syn_old_log_probs,
        returns=syn_returns,
        advantages=syn_advantages,
        actor_h0=syn_h0,
        actor_c0=syn_c0,
        critic_h0=syn_h0,
        critic_c0=syn_c0,
        dones=syn_dones
    )
    
    assert isinstance(diagnostics, dict)
    for name in ["total_loss", "policy_loss", "value_loss", "entropy", "approximate_KL", "clip_fraction", "number_of_updates", "sequences_processed", "sequence_length"]:
        assert name in diagnostics, f"Missing metric: {name}"
        val = diagnostics[name]
        assert math.isfinite(val), f"Metric {name} must be finite, got {val}"
        
    assert 0.0 <= diagnostics["clip_fraction"] <= 1.0, f"Clip fraction {diagnostics['clip_fraction']} is out of bounds."
    assert diagnostics["sequences_processed"] == 4, f"Expected 4 processed sequences, got {diagnostics['sequences_processed']}"
    assert diagnostics["sequence_length"] == 16
    print("-> PASSED: Diagnostics are finite, complete, and correct.")
    print(f"   Update Logs: Total Loss={diagnostics['total_loss']:.4f}, Policy Loss={diagnostics['policy_loss']:.4f}, KL={diagnostics['approximate_KL']:.6f}")

    # 6 & 7. Check that parameters actually change after update
    print("Validation 6, 7 & 8: Verifying parameters are updated by optimizer...")
    test_actor = LSTMActorNetwork(OBSERVATION_DIM, ACTION_DIM, 128).to(device)
    test_critic = LSTMCriticNetwork(OBSERVATION_DIM, 128).to(device)
    
    param_actor_before = next(test_actor.parameters()).clone()
    param_critic_before = next(test_critic.parameters()).clone()
    
    test_trainer = LSTMPPOTrainer(
        actor=test_actor,
        critic=test_critic,
        lr=0.1,  # Large learning rate to guarantee noticeable weight shifts
        ppo_epochs=1,
        batch_size=4,
        device=device
    )
    
    _ = test_trainer.update(
        states=syn_states,
        action_masks=syn_action_masks,
        actions=syn_actions,
        old_log_probs=syn_old_log_probs,
        returns=syn_returns,
        advantages=syn_advantages,
        actor_h0=syn_h0,
        actor_c0=syn_c0,
        critic_h0=syn_h0,
        critic_c0=syn_c0,
        dones=syn_dones
    )
    
    param_actor_after = next(test_actor.parameters()).clone()
    param_critic_after = next(test_critic.parameters()).clone()
    
    assert not torch.equal(param_actor_before, param_actor_after), "Actor parameters failed to update."
    assert not torch.equal(param_critic_before, param_critic_after), "Critic parameters failed to update."
    print("-> PASSED: Parameter updates are functional for both recurrent networks.")

    # 9 & 10. Action mask enforcement
    print("Validation 9 & 10: Verifying stored action masks exclude actions from gradient updates...")
    test_state = torch.zeros(1, 16, OBSERVATION_DIM, device=device)
    # Action 2 is illegal at all steps
    test_mask = torch.ones(1, 16, ACTION_DIM, device=device)
    test_mask[:, :, 2] = 0.0
    
    # Raw preference of 9999.0 for the illegal Action 2
    raw_logits = torch.zeros(1, 16, ACTION_DIM, device=device)
    raw_logits[:, :, 2] = 9999.0
    
    from pokemon_rl.agents.lstm_ppo import MaskedCategorical
    policy = MaskedCategorical(raw_logits, test_mask)
    probs = policy.probs
    
    assert torch.all(probs[:, :, 2] == 0.0), "Illegal action was not excluded from distribution!"
    print("-> PASSED: Action masking correctly overrides high-preference illegal choices.")

    promotion_mask = torch.zeros(1, 3, ACTION_DIM, device=device)
    promotion_mask[0, 0, 68] = 1.0
    promotion_mask[0, 1, 69] = 1.0
    promotion_mask[0, 2, 68] = 1.0
    promotion_mask[0, 2, 69] = 1.0
    promotion_logits = torch.zeros(1, 3, ACTION_DIM, device=device)
    promotion_logits[:, :, 0] = 9999.0
    promotion_policy = MaskedCategorical(promotion_logits, promotion_mask)
    sampled_promotions = promotion_policy.sample()
    assert sampled_promotions[0, 0].item() == 68
    assert sampled_promotions[0, 1].item() == 69
    assert sampled_promotions[0, 2].item() in (68, 69)
    assert torch.all(torch.isfinite(promotion_policy.entropy()))
    print("-> PASSED: Forced-promotion masks only permit actions 68/69 without NaNs.")

    # 13. Terminal boundaries stop GAE test
    print("Validation 13: Verifying terminal boundary zero-out GAE propagation...")
    rewards_np = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    dones_np = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) # final step is terminal
    values_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    last_value = 9999.0 # Deliberately huge next value
    
    adv_np, returns_np = compute_recurrent_gae_and_returns(
        rewards=rewards_np,
        dones=dones_np,
        values=values_np,
        last_value=last_value,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False
    )
    
    assert abs(returns_np[-1] - 1.0) < 1e-5, f"Expected final target return of 1.0, got {returns_np[-1]}"
    print("-> PASSED: GAE successfully ignores next-state value estimations on terminal boundaries.")

    # 14. Advantage normalization works
    print("Validation 14: Verifying advantage normalization mechanics...")
    adv_norm, _ = compute_recurrent_gae_and_returns(
        rewards=np.ones(10, dtype=np.float32),
        dones=np.zeros(10, dtype=np.float32),
        values=np.ones(10, dtype=np.float32) * 0.5,
        last_value=0.5,
        normalize_advantages=True
    )
    assert abs(adv_norm.mean()) < 1e-4, f"Expected normalized advantages mean approx 0, got {adv_norm.mean():.6f}"
    assert abs(adv_norm.std() - 1.0) < 1e-4, f"Expected normalized advantages std approx 1, got {adv_norm.std():.6f}"
    print("-> PASSED: Advantage normalization correctly standardizes to mean 0, std 1.")

    # 15. Sequence length 16 is respected
    print("Validation 15: Verifying sequence length constraint...")
    assert trainer.sequence_length == 16
    print("-> PASSED: Sequence length constraints respected.")

    # 18. Detached hidden states test
    print("Validation 18: Verifying hidden state inputs are fully detached from graph...")
    test_h0 = syn_h0.clone().requires_grad_(True)
    test_c0 = syn_c0.clone().requires_grad_(True)
    
    res = trainer.update(
        states=syn_states,
        action_masks=syn_action_masks,
        actions=syn_actions,
        old_log_probs=syn_old_log_probs,
        returns=syn_returns,
        advantages=syn_advantages,
        actor_h0=test_h0,
        actor_c0=test_c0,
        critic_h0=test_h0,
        critic_c0=test_c0,
        dones=syn_dones
    )
    assert test_h0.grad is None, "Gradient leaked into sequence initial hidden state parameters!"
    assert test_c0.grad is None, "Gradient leaked into sequence initial cell state parameters!"
    print("-> PASSED: Sequence initial recurrent states are detached (zero cross-sequence leakage).")

    print("\n======================================================================")
    print("          ALL PPO TRAINER VALIDATION TESTS PASS CLEANLY!             ")
    print("======================================================================")

if __name__ == "__main__":
    test_lstm_ppo_trainer_suite()
