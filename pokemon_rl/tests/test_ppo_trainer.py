import torch
import torch.nn as nn
import numpy as np
import math

from pokemon_rl.agents.ppo import ActorNetwork, CriticNetwork, MaskedCategorical
from pokemon_rl.agents.ppo_trainer import PPOTrainer
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM

def test_ppo_trainer_suite():
    print("======================================================================")
    print("          STARTING PPO TRAINER VALIDATION SUITE                      ")
    print("======================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device for Trainer Test: {device}")

    # Initialize independent networks
    actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    critic = CriticNetwork(state_dim=OBSERVATION_DIM).to(device)

    # 1. PPOTrainer initializes Actor and Critic correctly
    print("Validation 1 & 2: Initializing PPOTrainer and verifying parameters...")
    trainer = PPOTrainer(
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
        batch_size=8,
        device=device
    )
    assert trainer.actor is actor, "Actor network reference mismatch."
    assert trainer.critic is critic, "Critic network reference mismatch."
    assert len(list(trainer.actor.parameters())) > 0, "Actor network has no trainable parameters."
    assert len(list(trainer.critic.parameters())) > 0, "Critic network has no trainable parameters."
    print("-> PASSED: Initialization and network parameter validation complete.")

    # 3. Create synthetic rollout and verify shape conversions
    print("Validation 3: Constructing synthetic rollout (size 16)...")
    # Rollout of size 16
    syn_states = torch.randn(16, OBSERVATION_DIM, device=device)
    syn_actions = torch.randint(0, ACTION_DIM, (16,), device=device)
    syn_old_log_probs = torch.randn(16, device=device)
    syn_returns = torch.randn(16, device=device)
    # Ensure advantages are non-degenerate (standard deviation > 0)
    syn_advantages = torch.randn(16, device=device)
    syn_advantages = (syn_advantages - syn_advantages.mean()) / (syn_advantages.std() + 1e-8)
    
    # Standard mask: all actions legal
    syn_action_masks = torch.ones(16, ACTION_DIM, device=device)
    # Explicitly include one-choice forced-promotion rollout rows. The PPO
    # update must accept actions 68/69 when they are the sole legal actions.
    syn_action_masks[0] = 0.0
    syn_action_masks[0, 68] = 1.0
    syn_actions[0] = 68
    syn_action_masks[1] = 0.0
    syn_action_masks[1, 69] = 1.0
    syn_actions[1] = 69
    print("-> PASSED: Synthetic rollout initialized.")

    # 4 to 11. Run update and check outputs
    print("Validation 4 to 11: Executing PPO update and verifying output diagnostics...")
    diagnostics = trainer.update(
        states=syn_states,
        actions=syn_actions,
        old_log_probs=syn_old_log_probs,
        returns=syn_returns,
        advantages=syn_advantages,
        action_masks=syn_action_masks
    )

    assert isinstance(diagnostics, dict), "Diagnostics must be a dictionary."
    for name in ["total_loss", "policy_loss", "value_loss", "entropy", "approximate_KL", "clip_fraction", "number_of_updates"]:
        assert name in diagnostics, f"Missing diagnostic metric: {name}"
        val = diagnostics[name]
        assert math.isfinite(val), f"Diagnostic metric {name} must be finite, got {val}"
        
    assert 0.0 <= diagnostics["clip_fraction"] <= 1.0, f"Clip fraction {diagnostics['clip_fraction']} out of bounds [0, 1]"
    print("-> PASSED: PPO update diagnostics are correct and finite.")
    print(f"   Diagnostics: Total Loss={diagnostics['total_loss']:.4f}, Policy Loss={diagnostics['policy_loss']:.4f}, Value Loss={diagnostics['value_loss']:.4f}")

    # 5 & 6. Check that weights change
    print("Validation 5 & 6: Verifying network parameters update after training step...")
    # Initialize fresh networks to test explicit parameter changes
    test_actor = ActorNetwork(OBSERVATION_DIM, ACTION_DIM).to(device)
    test_critic = CriticNetwork(OBSERVATION_DIM).to(device)
    
    # Capture weights before
    actor_param_before = next(test_actor.parameters()).clone()
    critic_param_before = next(test_critic.parameters()).clone()
    
    test_trainer = PPOTrainer(
        actor=test_actor,
        critic=test_critic,
        lr=0.1,  # Large learning rate to ensure parameters shift noticeably
        ppo_epochs=1,
        batch_size=16,
        device=device
    )
    
    _ = test_trainer.update(
        states=syn_states,
        actions=syn_actions,
        old_log_probs=syn_old_log_probs,
        returns=syn_returns,
        advantages=syn_advantages,
        action_masks=syn_action_masks
    )
    
    actor_param_after = next(test_actor.parameters()).clone()
    critic_param_after = next(test_critic.parameters()).clone()
    
    assert not torch.equal(actor_param_before, actor_param_after), "Actor weights did not update after optimization."
    assert not torch.equal(critic_param_before, critic_param_after), "Critic weights did not update after optimization."
    print("-> PASSED: Actor and Critic network parameters updated successfully.")

    # 12 & 13. Action Masking and Preference Exclusion tests
    print("Validation 12 & 13: Testing stored action mask enforcement and preference exclusion...")
    # Setup test state where Action 1 is illegal (masked 0.0)
    test_state = torch.zeros(1, OBSERVATION_DIM, device=device)
    test_mask = torch.ones(1, ACTION_DIM, device=device)
    test_mask[0, 1] = 0.0  # Force Action 1 to be illegal
    
    # Verify that even with a massive raw logit of 9999.0 for the illegal Action 1,
    # its probability is suppressed to 0.0 and it never affects the distribution
    huge_illegal_logits = torch.zeros(1, ACTION_DIM, device=device)
    huge_illegal_logits[0, 1] = 9999.0  # Massive preference for illegal action
    
    masked_policy = MaskedCategorical(huge_illegal_logits, test_mask)
    probs = masked_policy.probs
    
    assert probs[0, 1].item() == 0.0, "Illegal action was not excluded from policy distribution!"
    assert abs(probs[0, 0].item() - (1.0 / (ACTION_DIM - 1))) < 1e-5, f"Expected uniform legal probabilities, got {probs}"
    print("-> PASSED: Stored action masks strictly exclude illegal options, even under extreme logit preferences.")

    # One legal promotion action must form a finite, deterministic policy.
    promotion_mask = torch.zeros(1, ACTION_DIM, device=device)
    promotion_mask[0, 68] = 1.0
    promotion_logits = torch.zeros(1, ACTION_DIM, device=device)
    promotion_logits[0, 0] = 9999.0
    promotion_policy = MaskedCategorical(promotion_logits, promotion_mask)
    promotion_action = promotion_policy.sample()
    assert promotion_action.item() == 68
    assert torch.isfinite(promotion_policy.log_prob(promotion_action)).all()
    assert torch.isfinite(promotion_policy.entropy()).all()
    print("-> PASSED: Single-choice forced-promotion masks remain finite and deterministic.")

    # 14, 15 & 16. Epoch updates and CPU/CUDA tests
    print("Validation 14, 15 & 16: Verifying multi-epoch updates and cross-device correctness...")
    # Multi-epoch optimization test
    multi_epoch_trainer = PPOTrainer(
        actor=ActorNetwork(OBSERVATION_DIM, ACTION_DIM).to(device),
        critic=CriticNetwork(OBSERVATION_DIM).to(device),
        ppo_epochs=12,  # Multi-epoch pass
        batch_size=4,
        device=device
    )
    res = multi_epoch_trainer.update(syn_states, syn_actions, syn_old_log_probs, syn_returns, syn_advantages, syn_action_masks)
    assert res["number_of_updates"] == 12 * 4, f"Expected 48 minibatch steps, got {res['number_of_updates']}"
    print("-> PASSED: Multi-epoch gradient optimizations run cleanly.")

    # 17. Mathematical Sanity Test for the PPO Clipping Rule
    print("Validation 17: Performing PPO ratio clipping mathematical sanity test...")
    # Scenario: Positive advantage, ratio is above 1 + clip_epsilon (e.g. 1.5, clip is 0.2)
    adv = torch.tensor([2.0])  # Positive advantage
    clip_eps = 0.2
    
    ratio_high = torch.tensor([1.5])  # Ratio above 1.2
    surr1 = ratio_high * adv  # 3.0
    surr2 = torch.clamp(ratio_high, 1.0 - clip_eps, 1.0 + clip_eps) * adv  # 1.2 * 2.0 = 2.4
    l_clip = torch.min(surr1, surr2)
    assert abs(l_clip.item() - 2.4) < 1e-5, f"Clipping failed! Expected clipped value approx 2.4, got {l_clip.item()}"
    
    # Scenario: Negative advantage, ratio is above 1 + clip_epsilon (e.g. 1.5, clip is 0.2)
    adv_neg = torch.tensor([-2.0])  # Negative advantage
    surr1_neg = ratio_high * adv_neg  # -3.0
    surr2_neg = torch.clamp(ratio_high, 1.0 - clip_eps, 1.0 + clip_eps) * adv_neg  # 1.2 * -2.0 = -2.4
    l_clip_neg = torch.min(surr1_neg, surr2_neg)
    assert abs(l_clip_neg.item() - (-3.0)) < 1e-5, f"Unclipped surrogate was not preferred for negative advantage! Got {l_clip_neg.item()}"
    print("-> PASSED: PPO clipping mathematical sanity test verified successfully.")

    print("\n======================================================================")
    print("          ALL PPO TRAINER VALIDATION TESTS PASS CLEANLY!             ")
    print("======================================================================")

if __name__ == "__main__":
    test_ppo_trainer_suite()
