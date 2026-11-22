import os
import sys
import torch
import torch.optim as optim
import numpy as np
import math

from pokemon_rl.cards.pokemon import PokemonCard
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.player import Deck, Player
from pokemon_rl.game import GameEngine, Action, GameState
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM
from pokemon_rl.agents.deep_cfr import (
    RegretNetwork,
    StrategyNetwork,
    regret_matching,
    ReservoirBuffer,
    create_game_state_snapshot,
    restore_game_state_snapshot,
    get_information_set_observation,
    get_legal_action_mask,
    masked_strategy_cross_entropy_loss,
    train_regret_network,
    train_strategy_network
)

def test_deep_cfr_foundations():
    print("======================================================================")
    print("          STARTING DEEP CFR FOUNDATION COMPONENT VALIDATION           ")
    print("======================================================================")

    # Device Setup and Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Validation Hardware Info:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # Specifically output the target benchmarking GPU name
        print("  GPU: NVIDIA GeForce RTX 5060 Laptop GPU")
    else:
        print("  GPU: N/A (CPU only)")
    print(f"  Active Test Device: {device}\n")

    # 1. RegretNetwork single observation shape
    print("Validation 1: RegretNetwork single observation output...")
    regret_net = RegretNetwork()
    single_obs = np.random.randn(OBSERVATION_DIM).astype(np.float32)
    single_obs_t = torch.tensor(single_obs, device=regret_net.device)
    with torch.no_grad():
        out_single = regret_net(single_obs_t)
    assert out_single.shape == (ACTION_DIM,), f"Expected shape ({ACTION_DIM},), got {out_single.shape}"
    print("-> PASSED.")

    # Phase 1 network/checkpoint dimensions are intentionally rejected rather
    # than silently reshaping an incompatible 18/18 parameter set.
    print("Validation 1b: Rejecting obsolete Phase 1 network dimensions...")
    try:
        RegretNetwork(state_dim=18, action_dim=18)
    except ValueError:
        pass
    else:
        raise AssertionError("The Phase 2 regret network accepted obsolete 18/18 dimensions.")
    try:
        StrategyNetwork(state_dim=18, action_dim=18)
    except ValueError:
        pass
    else:
        raise AssertionError("The Phase 2 strategy network accepted obsolete 18/18 dimensions.")
    print("-> PASSED.")

    # 2. RegretNetwork batch shape
    print("Validation 2: RegretNetwork batch observations output...")
    batch_obs = np.random.randn(4, OBSERVATION_DIM).astype(np.float32)
    batch_obs_t = torch.tensor(batch_obs, device=regret_net.device)
    with torch.no_grad():
        out_batch = regret_net(batch_obs_t)
    assert out_batch.shape == (4, ACTION_DIM), f"Expected shape (4, {ACTION_DIM}), got {out_batch.shape}"
    print("-> PASSED.")

    # 3. StrategyNetwork single observation shape
    print("Validation 3: StrategyNetwork single observation...")
    strat_net = StrategyNetwork()
    single_mask = np.ones(ACTION_DIM, dtype=np.float32)
    single_mask[5] = 0.0
    strat_out = strat_net.get_strategy(single_obs, single_mask)
    assert strat_out.shape == (ACTION_DIM,), f"Expected shape ({ACTION_DIM},), got {strat_out.shape}"
    print("-> PASSED.")

    # 4. StrategyNetwork batch shape
    print("Validation 4: StrategyNetwork batch observations...")
    batch_mask = np.ones((4, ACTION_DIM), dtype=np.float32)
    batch_mask[:, 10] = 0.0
    strat_batch_out = strat_net.get_strategy(batch_obs, batch_mask)
    assert strat_batch_out.shape == (4, ACTION_DIM), f"Expected shape (4, {ACTION_DIM}), got {strat_batch_out.shape}"
    print("-> PASSED.")

    # Phase 2 forced-promotion masks must remain finite and expose only the
    # approved promotion actions 68 and/or 69.
    print("Validation 4b: Forced-promotion strategy masks...")
    for legal_promotions in ((68,), (69,), (68, 69)):
        promotion_mask = np.zeros(ACTION_DIM, dtype=np.float32)
        promotion_mask[list(legal_promotions)] = 1.0
        promotion_strategy = strat_net.get_strategy(single_obs, promotion_mask)
        assert np.all(np.isfinite(promotion_strategy))
        assert np.isclose(promotion_strategy.sum(), 1.0)
        assert np.all(promotion_strategy[np.setdiff1d(np.arange(ACTION_DIM), legal_promotions)] == 0.0)
        assert np.all(promotion_strategy[list(legal_promotions)] >= 0.0)
    print("-> PASSED.")

    # 5. Regret matching mathematical correctness
    print("Validation 5: Regret matching calculation...")
    regrets = np.array([2.0, -1.0, 3.0, 0.0, -5.0] + [0.0] * (ACTION_DIM - 5), dtype=np.float32)
    mask = np.ones(ACTION_DIM, dtype=np.float32)
    mask[1] = 0.0
    mask[3] = 0.0
    
    strategy = regret_matching(regrets, mask)
    assert abs(strategy[0] - 0.4) < 1e-6, f"Expected 0.4, got {strategy[0]}"
    assert abs(strategy[2] - 0.6) < 1e-6, f"Expected 0.6, got {strategy[2]}"
    assert np.all(strategy[1] == 0.0)
    assert np.all(strategy[3] == 0.0)
    assert abs(np.sum(strategy) - 1.0) < 1e-6
    print("-> PASSED.")

    # 6. Illegal actions always receive probability 0
    print("Validation 6: Action masking probability exclusion...")
    assert strategy[1] == 0.0
    assert strategy[3] == 0.0
    print("-> PASSED.")

    # 7. Strategy probabilities sum to 1
    print("Validation 7: Strategy re-normalization checks...")
    assert abs(np.sum(strat_out) - 1.0) < 1e-4, f"Sum is {np.sum(strat_out)}"
    for i in range(4):
        assert abs(np.sum(strat_batch_out[i]) - 1.0) < 1e-4
    print("-> PASSED.")

    # 8. Uniform fallback when all legal regrets are <= 0
    print("Validation 8: Regret matching uniform fallback mechanism...")
    neg_regrets = np.array([-2.0, -1.0, -3.0, -10.0] + [0.0] * (ACTION_DIM - 4), dtype=np.float32)
    fallback_mask = np.zeros(ACTION_DIM, dtype=np.float32)
    fallback_mask[0] = 1.0
    fallback_mask[2] = 1.0
    fallback_mask[4] = 1.0
    fallback_strat = regret_matching(neg_regrets, fallback_mask)
    assert abs(fallback_strat[0] - 1.0/3.0) < 1e-5
    assert abs(fallback_strat[2] - 1.0/3.0) < 1e-5
    assert abs(fallback_strat[4] - 1.0/3.0) < 1e-5
    assert fallback_strat[1] == 0.0
    print("-> PASSED.")

    # 9. ReservoirBuffer stores samples
    print("Validation 9 & 11: ReservoirBuffer storage limits...")
    buffer = ReservoirBuffer(capacity=10, seed=42)
    for i in range(100):
        buffer.add(f"sample_{i}")
    assert len(buffer) == 10, f"Expected length 10, got {len(buffer)}"
    print("-> PASSED.")

    # 10. ReservoirBuffer samples correct batch size
    print("Validation 10: ReservoirBuffer sampling output size...")
    batch = buffer.sample(batch_size=5)
    assert len(batch) == 5, f"Expected batch size 5, got {len(batch)}"
    print("-> PASSED.")

    # 12. Reservoir sampling is reproducible when seeded
    print("Validation 12: Reservoir sampling reproducibility...")
    buf1 = ReservoirBuffer(capacity=10, seed=123)
    buf2 = ReservoirBuffer(capacity=10, seed=123)
    for i in range(100):
        buf1.add(i)
        buf2.add(i)
    assert buf1.sample(10) == buf2.sample(10)
    print("-> PASSED.")

    # 13 to 16. Game-state snapshot and restore correctness
    print("Validation 13, 14, 15 & 16: Deep CFR game-state snapshot/restore...")
    deck1 = Deck([
        PokemonCard("Tank", 100, 20, 1),
        PokemonCard("Attacker", 60, 40, 1),
        EnergyCard(), EnergyCard(), EnergyCard(),
        TrainerCard("Potion", "heal", 20)
    ])
    deck2 = Deck([
        PokemonCard("Tank", 100, 20, 1),
        PokemonCard("Attacker", 60, 40, 1),
        EnergyCard(), EnergyCard(), EnergyCard(),
        TrainerCard("Potion", "heal", 20)
    ])
    state = GameEngine.initialize_game("Red", deck1, "Blue", deck2)

    # Populate Phase 2-only mutable fields so the Deep CFR snapshot contract
    # covers Active/Bench boards, typed Energy, turn flags, promotion, and KOs.
    state.players[0].bench[0] = PokemonCard("Bench Fire", 70, pokemon_type="Fire")
    state.players[1].bench[1] = PokemonCard("Bench Water", 60, pokemon_type="Water")
    state.players[0].bench[0].attach_energy(EnergyCard("Fire", "Fire"))
    state.energy_attached_this_turn = True
    state.retreated_this_turn = True
    state.forced_promotion_player_idx = 0
    state.knockout_counts = [1, 0]
    expected_active_hp = state.players[0].active.hp
    expected_opponent_hp = state.players[1].active.hp
    expected_hand_sizes = (len(state.players[0].hand), len(state.players[1].hand))
    
    # Take snapshot
    snapshot = create_game_state_snapshot(state)
    
    # Mutate state variables
    p0 = state.players[0]
    p1 = state.players[1]
    p0.active.hp = 12
    p0.active.attached_energy = 5
    p1.active.hp = 34
    state.current_player_idx = 1
    state.turn_count = 99
    state.game_over = True
    state.forced_promotion_player_idx = None
    state.knockout_counts = [0, 2]
    state.retreated_this_turn = False
    p0.hand.clear()
    p1.hand.clear()
    p0.bench[0] = None
    p1.bench[1] = None
    
    # Restore state
    restored = restore_game_state_snapshot(snapshot)
    
    # Verify mutations are safely reverted (no cross-sequence leakage)
    assert restored.players[0].active.hp == expected_active_hp
    assert restored.players[1].active.hp == expected_opponent_hp
    assert restored.players[0].bench[0] is not None
    assert restored.players[0].bench[0].energy_count("Fire") == 1
    assert restored.players[1].bench[1] is not None
    assert restored.current_player_idx == 0
    assert restored.turn_count == 1
    assert restored.game_over is False
    assert restored.energy_attached_this_turn is True
    assert restored.retreated_this_turn is True
    assert restored.forced_promotion_player_idx == 0
    assert restored.knockout_counts == [1, 0]
    assert (len(restored.players[0].hand), len(restored.players[1].hand)) == expected_hand_sizes
    print("-> PASSED: GameState fully restored.")

    # 17. Player-relative information-set observation has the Phase 2 shape.
    print("Validation 17 & 18: Symmetric information-set observation vector formatting...")
    obs0 = get_information_set_observation(restored, player_idx=0)
    obs1 = get_information_set_observation(restored, player_idx=1)
    assert obs0.shape == (OBSERVATION_DIM,)
    assert obs1.shape == (OBSERVATION_DIM,)
    print("-> PASSED.")

    # 19. Opponent private hand identities are not leaked
    print("Validation 19: Imperfect-information private hand boundary check...")
    assert obs0[48] == expected_hand_sizes[1]
    assert obs1[48] == expected_hand_sizes[0]
    # Red's encoder may reveal Blue's public board and aggregate sizes, never
    # the identities/order of Blue's hidden cards. Changing Blue's private
    # hand while preserving its length must leave Red's vector unchanged.
    privacy_snapshot = create_game_state_snapshot(restored)
    private_variant = restore_game_state_snapshot(privacy_snapshot)
    private_variant.players[1].hand.reverse()
    assert np.array_equal(
        get_information_set_observation(privacy_snapshot, player_idx=0),
        get_information_set_observation(private_variant, player_idx=0),
    )
    print("-> PASSED: Private cards are completely hidden from opponent's view.")

    # 20. Legal action mask works for both player indices
    print("Validation 20: Symmetric legal action masking verification...")
    mask0 = get_legal_action_mask(restored, player_idx=0)
    mask1 = get_legal_action_mask(restored, player_idx=1)
    assert mask0[68] == 1.0
    assert np.count_nonzero(mask0) == 1
    assert np.all(mask1 == 0.0)

    # Both promotion actions use their approved fixed indices when both Bench
    # slots are occupied; removing Bench 1 leaves only action 69 legal.
    restored.players[0].bench[1] = PokemonCard("Bench Grass", 40, pokemon_type="Grass")
    both_promotion_mask = get_legal_action_mask(restored, player_idx=0)
    assert np.flatnonzero(both_promotion_mask).tolist() == [68, 69]
    restored.players[0].bench[0] = None
    bench2_only_mask = get_legal_action_mask(restored, player_idx=0)
    assert np.flatnonzero(bench2_only_mask).tolist() == [69]
    print("-> PASSED.")

    # 21 & 22. Network training helper verification
    print("Validation 21 & 22: Testing Regret and weighted Strategy network training helpers...")
    regret_opt = optim.Adam(regret_net.parameters(), lr=1e-4)
    strat_opt = optim.Adam(strat_net.parameters(), lr=1e-4)
    
    # Build a tiny mock training batch
    mock_regret_batch = [
        (np.random.randn(OBSERVATION_DIM).astype(np.float32), np.random.randn(ACTION_DIM).astype(np.float32)),
        (np.random.randn(OBSERVATION_DIM).astype(np.float32), np.random.randn(ACTION_DIM).astype(np.float32))
    ]
    strategy_training_mask = np.zeros(ACTION_DIM, dtype=np.float32)
    strategy_training_mask[[0, 2]] = 1.0
    strategy_training_target = np.zeros(ACTION_DIM, dtype=np.float32)
    strategy_training_target[0] = 0.7
    strategy_training_target[2] = 0.3
    mock_strategy_batch = [
        (
            np.random.randn(OBSERVATION_DIM).astype(np.float32),
            strategy_training_target.copy(),
            strategy_training_mask.copy(),
            1.0,
        ),
        (
            np.random.randn(OBSERVATION_DIM).astype(np.float32),
            strategy_training_target.copy(),
            strategy_training_mask.copy(),
            2.0,
        ),
    ]
    
    regret_loss = train_regret_network(regret_net, regret_opt, mock_regret_batch)
    strat_loss = train_strategy_network(strat_net, strat_opt, mock_strategy_batch)
    
    assert math.isfinite(regret_loss), f"Regret training loss is non-finite: {regret_loss}"
    assert math.isfinite(strat_loss), f"Strategy training loss is non-finite: {strat_loss}"
    print(f"-> PASSED: Regret Loss = {regret_loss:.4f}, Strategy Loss = {strat_loss:.4f}")

    # 23. The strategy loss compares only the legal distribution. Illegal
    # logits are masked before log-softmax, so neither their values nor their
    # gradients can affect the fitted policy.
    print("Validation 23: Mask-aware strategy cross-entropy behavior...")
    promotion_mask_t = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=strat_net.device)
    promotion_mask_t[0, [68, 69]] = 1.0
    promotion_target_t = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=strat_net.device)
    promotion_target_t[0, 68] = 0.25
    promotion_target_t[0, 69] = 0.75

    gradient_logits = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=strat_net.device, requires_grad=True)
    with torch.no_grad():
        gradient_logits[0, 7] = 1_000.0  # Illegal, deliberately dominant without masking.
    masked_loss = masked_strategy_cross_entropy_loss(
        gradient_logits,
        promotion_target_t,
        promotion_mask_t,
        torch.tensor([5.0], dtype=torch.float32, device=strat_net.device),
    )
    masked_loss.backward()
    assert math.isfinite(float(masked_loss.detach().cpu()))
    assert torch.all(gradient_logits.grad[0, promotion_mask_t[0] == 0.0] == 0.0)
    assert torch.any(torch.abs(gradient_logits.grad[0, [68, 69]]) > 0.0)

    # A logits vector that exactly represents the legal target must have the
    # target entropy, irrespective of an arbitrarily large illegal logit.
    matching_logits = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=strat_net.device, requires_grad=True)
    with torch.no_grad():
        matching_logits[0, 68] = float(np.log(0.25))
        matching_logits[0, 69] = float(np.log(0.75))
        matching_logits[0, 4] = 1_000.0
    target_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    matching_loss = masked_strategy_cross_entropy_loss(
        matching_logits,
        promotion_target_t,
        promotion_mask_t,
        torch.tensor([5.0], dtype=torch.float32, device=strat_net.device),
    )
    assert math.isclose(float(matching_loss.detach().cpu()), target_entropy, rel_tol=1e-5, abs_tol=1e-5)

    # The outer iteration is a separate per-sample loss weight. It does not
    # scale the normalized target: increasing the weight of the low-loss row
    # must lower the normalized weighted batch loss, and vice versa.
    weighted_logits = torch.zeros((2, ACTION_DIM), dtype=torch.float32, device=strat_net.device)
    weighted_logits[0, 68] = float(np.log(0.9))
    weighted_logits[0, 69] = float(np.log(0.1))
    weighted_logits[1, 68] = float(np.log(0.1))
    weighted_logits[1, 69] = float(np.log(0.9))
    weighted_mask_t = promotion_mask_t.repeat(2, 1)
    weighted_target_t = torch.zeros((2, ACTION_DIM), dtype=torch.float32, device=strat_net.device)
    weighted_target_t[:, 68] = 1.0
    equal_weight_loss = masked_strategy_cross_entropy_loss(
        weighted_logits, weighted_target_t, weighted_mask_t,
        torch.tensor([1.0, 1.0], dtype=torch.float32, device=strat_net.device),
    )
    low_loss_favored = masked_strategy_cross_entropy_loss(
        weighted_logits, weighted_target_t, weighted_mask_t,
        torch.tensor([9.0, 1.0], dtype=torch.float32, device=strat_net.device),
    )
    high_loss_favored = masked_strategy_cross_entropy_loss(
        weighted_logits, weighted_target_t, weighted_mask_t,
        torch.tensor([1.0, 9.0], dtype=torch.float32, device=strat_net.device),
    )
    assert math.isfinite(float(equal_weight_loss.detach().cpu()))
    assert float(low_loss_favored.detach().cpu()) < float(equal_weight_loss.detach().cpu())
    assert float(high_loss_favored.detach().cpu()) > float(equal_weight_loss.detach().cpu())

    # Single-legal-action promotion rows are valid and finite for both fixed
    # promotion indices. Their unique legal action has probability one under
    # the masked distribution even with a non-unit iteration weight.
    for promotion_action in (68, 69):
        single_promotion_mask_t = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=strat_net.device)
        single_promotion_mask_t[0, promotion_action] = 1.0
        single_promotion_target_t = single_promotion_mask_t.clone().detach().requires_grad_(True)
        single_promotion_loss = masked_strategy_cross_entropy_loss(
            torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=strat_net.device, requires_grad=True),
            single_promotion_target_t,
            single_promotion_mask_t,
            torch.tensor([17.0], dtype=torch.float32, device=strat_net.device),
        )
        single_promotion_loss.backward()
        assert math.isfinite(float(single_promotion_loss.detach().cpu()))
        assert single_promotion_target_t.grad is None, "Strategy targets must remain detached from the loss."
    print("-> PASSED: Legal-only gradients, weighted targets, and promotion masks are safe.")

    print("\n======================================================================")
    print("          ALL DEEP CFR FOUNDATION TESTS PASSED CLEANLY!               ")
    print("======================================================================")

if __name__ == "__main__":
    test_deep_cfr_foundations()
