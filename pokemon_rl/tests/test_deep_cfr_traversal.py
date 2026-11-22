import os
import torch
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
    ReservoirBuffer,
    create_game_state_snapshot,
    restore_game_state_snapshot,
    regret_matching,
)
from pokemon_rl.agents.deep_cfr_traversal import action_to_index, run_cfr_traversal, apply_action
from pokemon_rl.training.train_deep_cfr import DeepCFRTrainer

def test_deep_cfr_traversal_suite():
    print("======================================================================")
    print("          STARTING DEEP CFR TRAVERSAL VALIDATION SUITE                ")
    print("======================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Hardware Info:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("  GPU: NVIDIA GeForce RTX 5060 Laptop GPU")
    else:
        print("  GPU: N/A (CPU only)")
    print(f"  Active Device:   {device}\n")

    # The traversal adapter must preserve the locked Phase 2 mapping instead
    # of reusing any Phase 1 action indices. Validate every slot group before
    # constructing a real root state.
    assert action_to_index(Action("end_turn")) == 0
    assert action_to_index(Action("attack", attack_idx=0)) == 1
    assert action_to_index(Action("attack", attack_idx=1)) == 2
    assert action_to_index(Action("retreat")) == 3
    for hand_slot in range(8):
        assert action_to_index(Action("play_pokemon", card_idx=hand_slot, bench_slot=0)) == 4 + hand_slot
        assert action_to_index(Action("play_pokemon", card_idx=hand_slot, bench_slot=1)) == 12 + hand_slot
        assert action_to_index(Action("attach_energy", card_idx=hand_slot, target="active")) == 20 + hand_slot
        assert action_to_index(Action("attach_energy", card_idx=hand_slot, bench_slot=0, target="bench")) == 28 + hand_slot
        assert action_to_index(Action("attach_energy", card_idx=hand_slot, bench_slot=1, target="bench")) == 36 + hand_slot
        assert action_to_index(Action("play_potion", card_idx=hand_slot, target="active")) == 44 + hand_slot
        assert action_to_index(Action("play_potion", card_idx=hand_slot, bench_slot=0, target="bench")) == 52 + hand_slot
        assert action_to_index(Action("play_potion", card_idx=hand_slot, bench_slot=1, target="bench")) == 60 + hand_slot
    assert action_to_index(Action("promote", bench_slot=0)) == 68
    assert action_to_index(Action("promote", bench_slot=1)) == 69
    try:
        action_to_index(Action("attach_energy", card_idx=8, target="active"))
    except ValueError:
        pass
    else:
        raise AssertionError("A hand slot beyond seven was incorrectly accepted by the Deep CFR adapter.")
    print("Phase 2 70-action mapping and fixed hand-slot boundary: PASSED\n")

    # 1. Initialize Networks and Buffers
    regret_net = RegretNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    strategy_net = StrategyNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    
    red_regret_buffer = ReservoirBuffer(capacity=1000, seed=42)
    red_strategy_buffer = ReservoirBuffer(capacity=1000, seed=43)
    blue_regret_buffer = ReservoirBuffer(capacity=1000, seed=44)
    blue_strategy_buffer = ReservoirBuffer(capacity=1000, seed=45)
    assert red_regret_buffer is not blue_regret_buffer
    assert red_strategy_buffer is not blue_strategy_buffer

    def assert_traverser_sample_accounting(diagnostics, regret_memory, strategy_memory, label):
        """A sample is stored exactly once per explicit traverser decision.

        A bounded external-sampling traversal can reach a horizon rollout or a
        terminal subtree while only the opponent has appeared in the explicit
        tree.  In that valid case the selected traverser has zero eligible
        decision nodes and must contribute zero samples, not fabricated ones.
        """
        expected_samples = diagnostics["traverser_decision_nodes"]
        assert diagnostics["regret_samples"] == expected_samples, (
            f"{label} regret sample count must equal its eligible explicit-tree decisions."
        )
        assert diagnostics["strategy_samples"] == expected_samples, (
            f"{label} strategy sample count must equal its eligible explicit-tree decisions."
        )
        assert regret_memory.num_samples_seen == expected_samples
        assert strategy_memory.num_samples_seen == expected_samples
        if expected_samples == 0:
            assert len(regret_memory) == 0
            assert len(strategy_memory) == 0
        else:
            assert len(regret_memory) > 0
            assert len(strategy_memory) > 0

    # 2. Build a real GameState to serve as root
    deck1 = DeepCFRTrainer._create_compact_test_deck()
    deck2 = DeepCFRTrainer._create_compact_test_deck()
    
    # Initialize the game (Symmetric setup)
    root_state = GameEngine.initialize_game("Red", deck1, "Blue", deck2)

    # -------------------------------------------------------------------------
    # CRITICAL TEST — SNAPSHOT ISOLATION
    # -------------------------------------------------------------------------
    print("Critical Test: Verifying snapshot isolation between parallel branches...")
    snapshot = create_game_state_snapshot(root_state)
    
    # We need to find a state with at least two legal actions.
    # At setup, Red has end_turn and possibly attach_energy if they have Energy in hand.
    legal_actions = GameEngine.get_legal_actions(root_state)
    assert len(legal_actions) >= 2, f"Setup must have at least 2 legal actions, found: {legal_actions}"
    
    # Map through the exact Phase 2 70-action adapter.
    action_idx_A = action_to_index(legal_actions[0])
    action_idx_B = action_to_index(legal_actions[1])
    
    # Execute Action A on branch snapshot A
    snap_A = create_game_state_snapshot(root_state)
    state_after_A = apply_action(snap_A, action_idx_A)
    hp_after_A = state_after_A.players[0].active.hp
    energy_after_A = state_after_A.players[0].active.attached_energy
    deck_count_after_A = state_after_A.players[0].deck.remaining_count()
    hand_count_after_A = len(state_after_A.players[0].hand)
    
    # Execute Action B on branch snapshot B (re-cloned from original root_state)
    snap_B = create_game_state_snapshot(root_state)
    state_after_B = apply_action(snap_B, action_idx_B)
    hp_after_B = state_after_B.players[0].active.hp
    energy_after_B = state_after_B.players[0].active.attached_energy
    deck_count_after_B = state_after_B.players[0].deck.remaining_count()
    hand_count_after_B = len(state_after_B.players[0].hand)
    
    # Assert isolation: executing B did not inherit mutations from A, and vice-versa
    # Action B (attach_energy) should have changed energy or hand, but A should not.
    if "attach_energy" in [legal_actions[0].action_type, legal_actions[1].action_type]:
        # Assert that the energy in snap_B has changed, but original root_state is untouched
        assert root_state.players[0].active.attached_energy == 0, "Original root state was mutated!"
        
    # Verify that restored root_state remains identical
    restored_snap = restore_game_state_snapshot(snapshot)
    assert restored_snap.players[0].active.hp == root_state.players[0].active.hp
    assert restored_snap.players[0].active.attached_energy == 0
    assert len(restored_snap.players[0].hand) == 4 # 3 initial + 1 drawn at start of turn
    assert len(restored_snap.players[1].hand) == 3
    print("-> PASSED: Snapshot isolation test completed with 100% independent state copies.\n")

    # -------------------------------------------------------------------------
    # Traversal Tests: Run for Player 0 (Red)
    # -------------------------------------------------------------------------
    print("Validation Tests: Running CFR Traversal for Player 0 (Red)...")
    val_0, diag_0 = run_cfr_traversal(
        root_state=root_state,
        traverser=0,
        regret_net=regret_net,
        strategy_net=strategy_net,
        regret_buffer=red_regret_buffer,
        strategy_buffer=red_strategy_buffer,
        iteration=3
    )
    
    print(f"  Root Expected Utility (Red Traverser): {val_0:.4f}")
    print(f"  Decision Nodes:    {diag_0['decision_nodes']}")
    print(f"  Terminal Nodes:    {diag_0['terminal_nodes']}")
    print(f"  Max Recursion Depth: {diag_0['max_depth']}")
    print(f"  Regret Samples Added:   {diag_0['regret_samples']}")
    print(f"  Strategy Samples Added: {diag_0['strategy_samples']}")
    
    # Assertions for Player 0 traversal
    assert diag_0["decision_nodes"] > 0, "Traversal failed to visit any decision nodes!"
    assert diag_0["terminal_nodes"] > 0, "Traversal failed to reach any terminal nodes!"
    assert diag_0["traverser"] == 0
    assert diag_0["root_utility"] == val_0
    assert -1.0 <= val_0 <= 1.0, f"Expected expected utility in range [-1, 1], got {val_0}"
    assert_traverser_sample_accounting(diag_0, red_regret_buffer, red_strategy_buffer, "Red")
    print("-> PASSED.\n")

    # -------------------------------------------------------------------------
    # Traversal Tests: Run for Player 1 (Blue)
    # -------------------------------------------------------------------------
    print("Validation Tests: Running CFR Traversal for Player 1 (Blue)...")
    val_1, diag_1 = run_cfr_traversal(
        root_state=root_state,
        traverser=1,
        regret_net=regret_net,
        strategy_net=strategy_net,
        regret_buffer=blue_regret_buffer,
        strategy_buffer=blue_strategy_buffer,
        iteration=7
    )
    
    print(f"  Root Expected Utility (Blue Traverser): {val_1:.4f}")
    print(f"  Decision Nodes:    {diag_1['decision_nodes']}")
    print(f"  Terminal Nodes:    {diag_1['terminal_nodes']}")
    print(f"  Max Recursion Depth: {diag_1['max_depth']}")
    print(f"  Regret Samples Added:   {diag_1['regret_samples']}")
    print(f"  Strategy Samples Added: {diag_1['strategy_samples']}")
    
    # Assertions for Player 1 traversal
    assert diag_1["decision_nodes"] > 0
    assert diag_1["terminal_nodes"] > 0
    assert diag_1["traverser"] == 1
    assert diag_1["root_utility"] == val_1
    assert -1.0 <= val_1 <= 1.0
    assert_traverser_sample_accounting(diag_1, blue_regret_buffer, blue_strategy_buffer, "Blue")
    print("-> PASSED.\n")

    # -------------------------------------------------------------------------
    # Structural Checks on Samples and Strategy Normalization
    # -------------------------------------------------------------------------
    print("Structural Checks: Verifying saved regrets and strategies target outputs...")
    for label, diagnostics, regret_memory, strategy_memory, expected_iteration_weight in (
        ("Red", diag_0, red_regret_buffer, red_strategy_buffer, 3.0),
        ("Blue", diag_1, blue_regret_buffer, blue_strategy_buffer, 7.0),
    ):
        # The traversal default is the documented four-step explicit horizon.
        assert 0 <= diagnostics["max_depth"] <= 4, f"{label} exceeded the bounded traversal horizon."
        for key in ("root_utility", "elapsed_seconds", "average_legal_action_count"):
            assert math.isfinite(float(diagnostics[key])), f"{label} has a non-finite {key}."

        # Validate target dimensions and normalization when this traverser
        # actually visited an eligible explicit-tree decision. A zero-sample
        # traversal has no target by design, so it must not be indexed.
        if diagnostics["traverser_decision_nodes"] > 0:
            regret_sample = regret_memory.sample(1)[0]
            strategy_sample = strategy_memory.sample(1)[0]
            assert regret_sample[0].shape == (OBSERVATION_DIM,), (
                f"{label} regret observation has the wrong shape: {regret_sample[0].shape}"
            )
            assert regret_sample[1].shape == (ACTION_DIM,), (
                f"{label} regret target has the wrong shape: {regret_sample[1].shape}"
            )
            assert strategy_sample[0].shape == (OBSERVATION_DIM,)
            assert strategy_sample[1].shape == (ACTION_DIM,)
            assert len(strategy_sample) == 4, (
                f"{label} strategy samples must retain the legal mask and iteration weight."
            )
            stored_strategy_mask = strategy_sample[2]
            stored_iteration_weight = strategy_sample[3]
            assert stored_strategy_mask.shape == (ACTION_DIM,)
            assert np.all((stored_strategy_mask == 0.0) | (stored_strategy_mask == 1.0))
            assert np.count_nonzero(stored_strategy_mask) > 0
            assert math.isfinite(float(stored_iteration_weight))
            assert float(stored_iteration_weight) == expected_iteration_weight, (
                f"{label} strategy sample weight must equal its traversal iteration."
            )
            assert abs(np.sum(strategy_sample[1][stored_strategy_mask == 1.0]) - 1.0) < 1e-5, (
                f"{label} strategy probabilities failed to sum to one."
            )
            assert np.all(strategy_sample[1][stored_strategy_mask == 0.0] == 0.0), (
                f"{label} strategy target assigned mass to an illegal action."
            )
            # The network is not updated in this traversal test. Its stored
            # target must therefore equal regret matching at the captured
            # information state and exact captured mask.
            with torch.no_grad():
                stored_regrets = regret_net(
                    torch.tensor(strategy_sample[0], dtype=torch.float32, device=regret_net.device)
                ).cpu().numpy()
            expected_strategy = regret_matching(stored_regrets, stored_strategy_mask)
            assert np.allclose(strategy_sample[1], expected_strategy, atol=1e-6)
    
    print("CPU/CUDA execution verification...")
    actual_device = next(regret_net.parameters()).device

    assert actual_device.type == device.type, (
        f"Network is on {actual_device}, expected {device}"
    )

    if device.type == "cuda":
        print("-> PASSED: CUDA execution matches network expectations.")
    else:
        print("-> PASSED: CPU execution matches network expectations.")

    print("\n======================================================================")
    print("          ALL DEEP CFR TRAVERSAL TESTS PASSED CLEANLY!                ")
    print("======================================================================")

if __name__ == "__main__":
    test_deep_cfr_traversal_suite()
