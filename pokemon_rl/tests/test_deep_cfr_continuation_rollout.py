"""Regression tests for legal, current-state Deep CFR continuation rollouts.

These tests exercise hands longer than the fixed eight-slot action interface.
In particular, an Energy in hand slot 8 must never be aliased to the Potion
action at slot 0 (which is fixed action index 44 in Phase 2).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import numpy as np
import torch

from pokemon_rl.agents.deep_cfr_traversal import (
    CFRDiagnostics,
    apply_action,
    legal_actions_and_mask,
    rollout_to_terminal,
)
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.game import GameEngine, GameState
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM
from pokemon_rl.training.train_deep_cfr import DeepCFRTrainer


class _MixedActionRegretNetwork(torch.nn.Module):
    """Choose an Energy then Potion when those cards occupy slot zero.

    Once that short mixed sequence has changed the hand, End Turn is preferred.
    This keeps the continuation deterministic while still exercising action
    remapping after card removal for both players.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        regrets = torch.zeros((observations.shape[0], ACTION_DIM), dtype=torch.float32, device=observations.device)
        first_hand_slot = observations[:, 18]
        energy_already_attached = observations[:, 27] == 1.0
        attach_condition = (first_hand_slot == 10.0) & ~energy_already_attached
        potion_condition = first_hand_slot == 9.0
        regrets[attach_condition, 20] = 1.0  # Colorless Energy in hand slot 0 -> Active.
        regrets[potion_condition, 44] = 1.0  # Potion in hand slot 0 -> Active.
        regrets[~(attach_condition | potion_condition), 0] = 1.0
        return regrets


def _mixed_hand(player_label: str) -> List[Any]:
    """Create nine mixed card slots, including an unrepresentable Energy slot 8."""
    return [
        EnergyCard(f"{player_label} Energy 0"),
        TrainerCard(f"{player_label} Potion 1", "heal", 20),
        EnergyCard(f"{player_label} Energy 2"),
        TrainerCard(f"{player_label} Potion 3", "heal", 20),
        EnergyCard(f"{player_label} Energy 4"),
        TrainerCard(f"{player_label} Potion 5", "heal", 20),
        EnergyCard(f"{player_label} Energy 6"),
        TrainerCard(f"{player_label} Potion 7", "heal", 20),
        EnergyCard(f"{player_label} Energy 8"),
    ]


def _make_mixed_state(seed: int) -> GameState:
    random.seed(seed)
    state = GameEngine.initialize_game(
        "Red", DeepCFRTrainer._create_default_deck(),
        "Blue", DeepCFRTrainer._create_default_deck(),
    )
    for player_idx, label in ((0, "Red"), (1, "Blue")):
        player = state.players[player_idx]
        player.hand = _mixed_hand(label)
        assert player.active is not None
        # Each Potion is legal at the start of each player's first continuation
        # turn, allowing the test to verify its type and hand-slot mapping.
        player.active.hp = player.active.max_hp - 20
    state.current_player_idx = 0
    state.energy_attached_this_turn = False
    return state


def _assert_trace_invariants(trace: Dict[str, Any]) -> None:
    """Verify trace state/mask/card agreement immediately before every step."""
    action_index = trace["selected_action_index"]
    action = trace["selected_action"]
    assert trace["observation"].shape == (OBSERVATION_DIM,)
    assert trace["legal_mask"].shape == (ACTION_DIM,)
    assert 0 <= action_index < ACTION_DIM
    assert trace["legal_mask"][action_index] == 1.0
    assert action_index in trace["legal_action_indices"]
    assert trace["regret_network_player"] == trace["current_player"]
    assert trace["regret_network"] == {"type": "_MixedActionRegretNetwork", "device": "cpu"}

    # ``play_potion`` is GameEngine's concrete name for the approved Phase 2
    # 44--67 use-Potion action group. A rollout may choose any currently legal
    # Phase 2 action when regret matching falls back to a uniform policy.
    assert action["action_type"] in {
        "end_turn", "attack", "retreat", "promote",
        "attach_energy", "play_pokemon", "play_potion",
    }
    slot = action["hand_slot"]
    selected_card = trace["selected_card"]
    if action["action_type"] == "attach_energy":
        assert slot is not None and 0 <= slot < 8
        assert selected_card is not None and selected_card["class_name"] == "EnergyCard"
        if action["target"] == "active":
            assert action_index == 20 + slot
        else:
            assert action["target"] == "bench" and action["bench_slot"] in {0, 1}
            assert action_index == (28 if action["bench_slot"] == 0 else 36) + slot
    elif action["action_type"] == "play_pokemon":
        assert slot is not None and 0 <= slot < 8
        assert action["bench_slot"] in {0, 1}
        assert selected_card is not None and selected_card["class_name"] == "PokemonCard"
        assert action_index == (4 if action["bench_slot"] == 0 else 12) + slot
    elif action["action_type"] == "play_potion":
        assert slot is not None and 0 <= slot < 8
        assert selected_card is not None
        assert selected_card["class_name"] == "TrainerCard"
        assert selected_card["effect_type"] == "heal"
        if action["target"] == "active":
            assert action_index == 44 + slot
        else:
            assert action["target"] == "bench" and action["bench_slot"] in {0, 1}
            assert action_index == (52 if action["bench_slot"] == 0 else 60) + slot
    else:
        assert slot is None and selected_card is None
        if action["action_type"] == "end_turn":
            assert action_index == 0
        elif action["action_type"] == "attack":
            assert action["attack_idx"] in {0, 1}
            assert action_index == 1 + action["attack_idx"]
        elif action["action_type"] == "retreat":
            assert action_index == 3
        else:
            assert action["action_type"] == "promote"
            assert action["bench_slot"] in {0, 1}
            assert action_index == 68 + action["bench_slot"]


def _assert_slot_eight_energy_is_not_aliased(state: GameState, player_idx: int) -> None:
    state.current_player_idx = player_idx
    state.energy_attached_this_turn = False
    engine_actions = GameEngine.get_legal_actions(state)
    # The immutable 70-action contract exposes exactly hand slots 0--7. A
    # ninth card may exist in a private hand, but it must not create an action
    # or alias to any representable slot/action group.
    assert all(action.card_idx != 8 for action in engine_actions)

    indices, mask = legal_actions_and_mask(state)
    # Index 44 means *Potion on Active from hand slot 0*, not Energy at slot 8.
    # Slot 0 is deliberately Energy here, so index 44 must be absent.
    assert mask[44] == 0.0 and 44 not in indices
    assert mask[20] == 1.0  # Energy at the actually representable slot 0 -> Active.


def test_continuation_rollout_uses_current_legal_cards_for_both_players() -> None:
    print("=" * 70)
    print("     STARTING DEEP CFR CONTINUATION LEGAL-ACTION REGRESSION TEST")
    print("=" * 70)

    initial_state = _make_mixed_state(seed=8_081)
    _assert_slot_eight_energy_is_not_aliased(initial_state, player_idx=0)
    _assert_slot_eight_energy_is_not_aliased(initial_state, player_idx=1)
    print("Nine-card hand mask retains slot semantics for Red and Blue: PASSED")

    # The immediate validator fails before GameEngine.step if a caller tries to
    # reuse Potion-on-Active index 44 for the unsupported slot-8 Energy.
    initial_state.current_player_idx = 0
    try:
        apply_action(initial_state, 44)
    except RuntimeError as error:
        assert "not legal" in str(error)
    else:
        raise AssertionError("Unsupported slot-8 Energy was aliased to a Potion action.")
    print("Pre-step validation rejects stale or aliased fixed action indices: PASSED")

    regret_networks = {0: _MixedActionRegretNetwork(), 1: _MixedActionRegretNetwork()}
    all_traces: List[Dict[str, Any]] = []
    utilities: List[float] = []
    # Two complete continuations execute both sides' changing hands repeatedly;
    # each player receives policy decisions from its own regret-network entry.
    for seed, traverser in ((8_101, 0), (8_102, 1)):
        state = _make_mixed_state(seed)
        trace: List[Dict[str, Any]] = []
        np.random.seed(seed)
        utility = rollout_to_terminal(
            state=state,
            traverser=traverser,
            regret_net=regret_networks,
            diagnostics=CFRDiagnostics(),
            trace_callback=trace.append,
        )
        assert state.game_over
        assert math.isfinite(utility) and -1.0 <= utility <= 1.0
        assert trace
        assert all(trace_entry["current_player"] in {0, 1} for trace_entry in trace)
        for trace_entry in trace:
            _assert_trace_invariants(trace_entry)
        all_traces.extend(trace)
        utilities.append(utility)

    selected_types = {trace["selected_action"]["action_type"] for trace in all_traces}
    assert {"attach_energy", "play_potion", "end_turn"}.issubset(selected_types)
    assert {trace["current_player"] for trace in all_traces} == {0, 1}
    # Energy and Potion removal changes the hand before future policy queries.
    hand_signatures = {
        tuple((card["class_name"], card["name"]) for card in trace["hand_before"])
        for trace in all_traces
    }
    assert len(hand_signatures) > 2
    assert all(math.isfinite(value) for value in utilities)
    print("Repeated Red/Blue terminal continuations keep masks, networks, and cards aligned: PASSED")

    print("=" * 70)
    print("          DEEP CFR CONTINUATION REGRESSION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_continuation_rollout_uses_current_legal_cards_for_both_players()
