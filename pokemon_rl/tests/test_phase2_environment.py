"""Focused Phase 2 rules, mask, observation, and snapshot regression tests."""

from __future__ import annotations

from collections import Counter
import copy
import random

import numpy as np

from pokemon_rl.agents.deep_cfr import create_game_state_snapshot, restore_game_state_snapshot
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.pokemon import PokemonCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.environment import (
    ACTION_DIM,
    HAND_CARD_TYPE_IDS,
    OBSERVATION_DIM,
    OBSERVATION_INDEX,
    PokemonTCGEnv,
    get_phase2_information_set_observation,
)
from pokemon_rl.game import Action, GameEngine, GameState


def _fresh_card(name: str):
    """Return an independent exact Phase 2 card from a fresh supplied deck."""
    return next(card for card in PokemonTCGEnv._create_default_deck().initial_cards if card.name == name)


def _energy(energy_type: str) -> EnergyCard:
    return EnergyCard(f"Basic {energy_type} Energy", energy_type)


def _fresh_state(seed: int = 99) -> GameState:
    random.seed(seed)
    state = GameEngine.initialize_game(
        "Red", PokemonTCGEnv._create_default_deck(),
        "Blue", PokemonTCGEnv._create_default_deck(),
    )
    for player, active_name in ((state.players[0], "Pikachu"), (state.players[1], "Charmander")):
        player.active = _fresh_card(active_name)
        player.bench = [None, None]
        player.hand = []
        player.trash = []
    state.current_player_idx = 0
    state.energy_attached_this_turn = False
    state.retreated_this_turn = False
    state.forced_promotion_player_idx = None
    state.knockout_counts = [0, 0]
    state.game_over = False
    state.winner_idx = None
    return state


def _assert_raises_value_error(callable_obj) -> None:
    try:
        callable_obj()
    except ValueError:
        return
    raise AssertionError("Expected ValueError was not raised.")


def test_phase2_deck_and_action_contract() -> None:
    deck = PokemonTCGEnv._create_default_deck()
    assert len(deck.initial_cards) == 20
    assert Counter(card.name for card in deck.initial_cards) == {
        "Pikachu": 1, "Charmander": 1, "Swinub": 1, "Applin": 1,
        "Basic Grass Energy": 3, "Basic Fire Energy": 3,
        "Basic Water Energy": 3, "Basic Lightning Energy": 3, "Potion": 4,
    }
    assert [card.card_id for card in deck.initial_cards if isinstance(card, PokemonCard)] == [948, 926, 800, 149]
    assert all(card.heal_amount == 30 and card.card_id == 1117 for card in deck.initial_cards if isinstance(card, TrainerCard))

    env = PokemonTCGEnv()
    assert env.action_space.n == ACTION_DIM == 70
    assert tuple(OBSERVATION_INDEX) == tuple(range(50))
    for action_idx in range(ACTION_DIM):
        assert env._action_to_index(env._index_to_action(action_idx)) == action_idx
    assert env._index_to_action(4) == Action("play_pokemon", card_idx=0, bench_slot=0)
    assert env._index_to_action(19) == Action("play_pokemon", card_idx=7, bench_slot=1)
    assert env._index_to_action(20) == Action("attach_energy", card_idx=0, target="active")
    assert env._index_to_action(43) == Action("attach_energy", card_idx=7, bench_slot=1, target="bench")
    assert env._index_to_action(44) == Action("play_potion", card_idx=0, target="active")
    assert env._index_to_action(67) == Action("play_potion", card_idx=7, bench_slot=1, target="bench")
    assert env._index_to_action(68) == Action("promote", bench_slot=0)
    assert env._index_to_action(69) == Action("promote", bench_slot=1)
    print("Exact 20-card deck and all 70 action mappings: PASSED")


def test_phase2_observation_layout_and_privacy() -> None:
    state = _fresh_state()
    red, blue = state.players
    red.active.hp = 61
    red.active.attach_energy(_energy("Grass"))
    red.active.attach_energy(_energy("Lightning"))
    red.bench = [_fresh_card("Applin"), _fresh_card("Swinub")]
    red.bench[0].hp = 23
    red.bench[0].attach_energy(_energy("Fire"))
    red.bench[1].hp = 54
    red.bench[1].attach_energy(_energy("Water"))
    red.bench[1].attach_energy(_energy("Water"))
    red.hand = [_fresh_card("Applin"), _fresh_card("Charmander"), _energy("Grass"), _energy("Fire"),
                _energy("Water"), _energy("Lightning"), _fresh_card("Potion"), _fresh_card("Potion")]
    red.deck.cards = [_energy("Grass")] * 5
    state.energy_attached_this_turn = True
    state.retreated_this_turn = True
    state.forced_promotion_player_idx = 0

    blue.active.hp = 72
    blue.active.attach_energy(_energy("Fire"))
    blue.bench = [_fresh_card("Pikachu"), None]
    blue.bench[0].hp = 44
    blue.bench[0].attach_energy(_energy("Lightning"))
    blue.hand = [_fresh_card("Potion"), _energy("Water"), _fresh_card("Swinub")]
    blue.deck.cards = [_energy("Fire")] * 4

    observation = get_phase2_information_set_observation(state, player_idx=0)
    expected = np.array([
        61, 1, 0, 0, 1, 4,
        23, 0, 1, 0, 0, 1,
        54, 0, 0, 2, 0, 3,
        1, 2, 5, 6, 7, 8, 9, 9,
        5, 1, 1, 1,
        72, 0, 1, 0, 0, 2,
        44, 0, 0, 0, 1, 4,
        0, 0, 0, 0, 0, 0,
        3, 4,
    ], dtype=np.float32)
    assert observation.shape == (OBSERVATION_DIM,)
    assert np.array_equal(observation, expected), observation

    # Bench absence is six explicit zero values, and opponent hand identities
    # never affect an observation so long as public counts remain unchanged.
    red.bench[1] = None
    assert np.array_equal(get_phase2_information_set_observation(state, 0)[12:18], np.zeros(6, dtype=np.float32))
    private_view_before = get_phase2_information_set_observation(state, 0)
    blue.hand = [_energy("Lightning"), _fresh_card("Applin"), _fresh_card("Potion")]
    blue.deck.cards = [_fresh_card("Pikachu")] * 4
    private_view_after = get_phase2_information_set_observation(state, 0)
    assert np.array_equal(private_view_before, private_view_after)
    assert HAND_CARD_TYPE_IDS["potion"] == 9.0
    print("All 50 observation indices, absent Bench encoding, and privacy: PASSED")


def test_phase2_bench_energy_potion_and_masks() -> None:
    state = _fresh_state()
    red = state.players[0]
    red.hand = [_fresh_card("Charmander"), _fresh_card("Swinub"), _fresh_card("Applin")]
    GameEngine.step(state, Action("play_pokemon", card_idx=0, bench_slot=0))
    assert red.bench[0].name == "Charmander" and len(red.hand) == 2
    GameEngine.step(state, Action("play_pokemon", card_idx=0, bench_slot=1))
    assert red.bench[1].name == "Swinub" and len(red.hand) == 1
    assert not any(action.action_type == "play_pokemon" for action in GameEngine.get_legal_actions(state))
    _assert_raises_value_error(lambda: GameEngine.step(state, Action("play_pokemon", card_idx=0, bench_slot=0)))

    invalid_state = _fresh_state()
    invalid_state.players[0].hand = [_energy("Grass")]
    _assert_raises_value_error(lambda: GameEngine.step(invalid_state, Action("play_pokemon", card_idx=0, bench_slot=0)))

    for target, bench_slot in (("active", None), ("bench", 0), ("bench", 1)):
        target_state = _fresh_state()
        target_player = target_state.players[0]
        target_player.bench = [_fresh_card("Applin"), _fresh_card("Swinub")]
        target_player.hand = [_energy("Lightning")]
        GameEngine.step(target_state, Action("attach_energy", card_idx=0, target=target, bench_slot=bench_slot))
        target_card = target_player.active if target == "active" else target_player.bench[bench_slot]
        assert target_card.energy_count("Lightning") == 1
        assert target_state.energy_attached_this_turn is True

    potion_state = _fresh_state()
    potion_player = potion_state.players[0]
    potion_player.bench = [_fresh_card("Applin"), _fresh_card("Swinub")]
    potion_player.bench[0].hp = 5
    potion_player.hand = [_fresh_card("Potion")]
    GameEngine.step(potion_state, Action("play_potion", card_idx=0, target="bench", bench_slot=0))
    assert potion_player.bench[0].hp == 35
    assert len(potion_player.trash) == 1 and isinstance(potion_player.trash[0], TrainerCard)

    env = PokemonTCGEnv()
    env.state = _fresh_state()
    env.state.players[0].hand = [_fresh_card("Applin"), _energy("Grass")]
    env_mask = env.get_action_mask()
    assert env_mask[4] == 1.0 and env_mask[12] == 1.0 and env_mask[20 + 1] == 1.0
    assert env_mask[28 + 1] == 0.0 and env_mask[68] == 0.0 and env_mask[69] == 0.0

    forced_state = _fresh_state()
    forced_state.players[0].active = None
    forced_state.players[0].bench = [_fresh_card("Applin"), _fresh_card("Swinub")]
    forced_state.forced_promotion_player_idx = 0
    forced_env = PokemonTCGEnv()
    forced_env.state = forced_state
    forced_mask = forced_env.get_action_mask()
    assert np.flatnonzero(forced_mask).tolist() == [68, 69]
    print("Bench capacity, hand removal, Energy/Potion targets, and masks: PASSED")


def test_phase2_retreat_rules() -> None:
    state = _fresh_state()
    red = state.players[0]
    red.active = _fresh_card("Pikachu")
    red.active.attach_energy(_energy("Lightning"))
    red.bench = [_fresh_card("Charmander"), _fresh_card("Swinub")]
    GameEngine.step(state, Action("retreat"))
    assert red.active.name == "Charmander"
    assert red.bench[0].name == "Pikachu" and red.bench[1].name == "Swinub"
    assert red.bench[0].attached_energy == 0
    assert len(red.trash) == 1 and isinstance(red.trash[0], EnergyCard)
    assert state.retreated_this_turn is True
    assert Action("retreat") not in GameEngine.get_legal_actions(state)
    _assert_raises_value_error(lambda: GameEngine.step(state, Action("retreat")))

    GameEngine.step(state, Action("end_turn"))
    GameEngine.step(state, Action("end_turn"))
    assert state.current_player_idx == 0 and state.retreated_this_turn is False

    no_energy = _fresh_state()
    no_energy.players[0].active = _fresh_card("Swinub")
    no_energy.players[0].active.attach_energy(_energy("Water"))
    no_energy.players[0].bench[0] = _fresh_card("Pikachu")
    assert Action("retreat") not in GameEngine.get_legal_actions(no_energy)
    no_bench = _fresh_state()
    no_bench.players[0].active.attach_energy(_energy("Lightning"))
    assert Action("retreat") not in GameEngine.get_legal_actions(no_bench)
    print("Retreat cost, deterministic Bench 1 selection, limits, and reset: PASSED")


def test_phase2_weakness_knockouts_and_promotion() -> None:
    fire_ko = _fresh_state()
    fire_ko.players[0].active = _fresh_card("Charmander")
    fire_ko.players[0].active.attach_energy(_energy("Fire"))
    fire_ko.players[0].active.attach_energy(_energy("Fire"))
    fire_ko.players[1].active = _fresh_card("Applin")
    GameEngine.step(fire_ko, Action("attack", attack_idx=0))
    assert fire_ko.players[1].trash[-1].name == "Applin" and fire_ko.players[1].trash[-1].hp == 0
    assert fire_ko.knockout_counts == [1, 0] and fire_ko.game_over and fire_ko.winner_idx == 0

    water_damage = _fresh_state()
    water_damage.players[0].active = _fresh_card("Swinub")
    water_damage.players[0].active.attach_energy(_energy("Water"))
    water_damage.players[0].active.attach_energy(_energy("Water"))
    water_damage.players[1].active = _fresh_card("Charmander")
    GameEngine.step(water_damage, Action("attack", attack_idx=1))
    assert water_damage.players[1].active.hp == 40  # Icy Snow x2 Water weakness.

    normal_damage = _fresh_state()
    normal_damage.players[0].active = _fresh_card("Pikachu")
    normal_damage.players[0].active.attach_energy(_energy("Lightning"))
    normal_damage.players[0].active.attach_energy(_energy("Lightning"))
    normal_damage.players[1].active = _fresh_card("Charmander")
    GameEngine.step(normal_damage, Action("attack", attack_idx=1))
    assert normal_damage.players[1].active.hp == 50  # Lightning is not Charmander's Water weakness.

    promotion = _fresh_state()
    promotion.players[0].active = _fresh_card("Charmander")
    promotion.players[0].active.attach_energy(_energy("Fire"))
    promotion.players[0].active.attach_energy(_energy("Fire"))
    promotion.players[1].active = _fresh_card("Applin")
    promotion.players[1].bench = [_fresh_card("Pikachu"), _fresh_card("Swinub")]
    GameEngine.step(promotion, Action("attack", attack_idx=0))
    assert promotion.knockout_counts == [1, 0]
    assert promotion.current_player_idx == 1 and promotion.forced_promotion_player_idx == 1
    assert GameEngine.get_legal_actions(promotion) == [Action("promote", bench_slot=0), Action("promote", bench_slot=1)]
    GameEngine.step(promotion, Action("promote", bench_slot=1))
    assert promotion.players[1].active.name == "Swinub" and promotion.players[1].bench[1] is None
    assert promotion.forced_promotion_player_idx is None
    assert Action("end_turn") in GameEngine.get_legal_actions(promotion)

    promotion.players[1].active.hp = 30
    GameEngine.step(promotion, Action("end_turn"))
    GameEngine.step(promotion, Action("attack", attack_idx=0))
    assert promotion.knockout_counts == [2, 0]
    assert promotion.game_over and promotion.winner_idx == 0
    print("Weakness, normal damage, KO count, forced promotion, and two-KO win: PASSED")


def test_phase2_bench_out_deck_out_snapshot_and_alias_safety() -> None:
    bench_out = _fresh_state()
    bench_out.players[0].active = _fresh_card("Charmander")
    bench_out.players[0].active.attach_energy(_energy("Fire"))
    bench_out.players[0].active.attach_energy(_energy("Fire"))
    bench_out.players[1].active = _fresh_card("Applin")
    GameEngine.step(bench_out, Action("attack", attack_idx=0))
    assert bench_out.game_over and bench_out.winner_idx == 0 and bench_out.knockout_counts == [1, 0]

    deck_out = _fresh_state()
    deck_out.players[1].deck.cards = []
    GameEngine.step(deck_out, Action("end_turn"))
    assert deck_out.game_over and deck_out.winner_idx == 0

    snapshot_state = _fresh_state()
    snapshot_state.players[0].active.hp = 37
    snapshot_state.players[0].active.attach_energy(_energy("Grass"))
    snapshot_state.players[0].bench = [_fresh_card("Charmander"), _fresh_card("Swinub")]
    snapshot_state.players[0].bench[0].hp = 45
    snapshot_state.players[0].bench[0].attach_energy(_energy("Fire"))
    snapshot_state.retreated_this_turn = True
    snapshot_state.forced_promotion_player_idx = 0
    snapshot_state.knockout_counts = [1, 0]
    snapshot = create_game_state_snapshot(snapshot_state)
    snapshot_state.players[0].active.hp = 1
    snapshot_state.players[0].bench = [None, None]
    snapshot_state.retreated_this_turn = False
    snapshot_state.forced_promotion_player_idx = None
    snapshot_state.knockout_counts = [0, 1]
    restored = restore_game_state_snapshot(snapshot)
    assert restored.players[0].active.hp == 37 and restored.players[0].active.energy_count("Grass") == 1
    assert restored.players[0].bench[0].name == "Charmander" and restored.players[0].bench[0].hp == 45
    assert restored.players[0].bench[0].energy_count("Fire") == 1 and restored.players[0].bench[1].name == "Swinub"
    assert restored.retreated_this_turn and restored.forced_promotion_player_idx == 0 and restored.knockout_counts == [1, 0]

    alias_state = _fresh_state()
    alias_state.players[0].hand = [_energy("Grass") for _ in range(9)]
    alias_env = PokemonTCGEnv()
    alias_env.state = alias_state
    alias_mask = alias_env.get_action_mask()
    assert alias_mask[20] == 1.0 and alias_mask[44] == 0.0
    assert alias_env._action_to_index(Action("attach_energy", card_idx=8, target="active")) is None
    _assert_raises_value_error(lambda: GameEngine.step(alias_state, Action("attach_energy", card_idx=8, target="active")))
    _assert_raises_value_error(lambda: GameEngine.step(alias_state, Action("play_potion", card_idx=0, target="active")))

    deterministic_env = PokemonTCGEnv()
    observation_one, _ = deterministic_env.reset(seed=7_777)
    mask_one = deterministic_env.get_action_mask()
    observation_two, _ = deterministic_env.reset(seed=7_777)
    assert np.array_equal(observation_one, observation_two)
    assert np.array_equal(mask_one, deterministic_env.get_action_mask())
    print("Bench-out, deck-out, snapshot/restore, slot validation, and deterministic reset: PASSED")


def run_phase2_environment_tests() -> None:
    print("=" * 70)
    print("            STARTING PHASE 2 ENVIRONMENT TEST SUITE")
    print("=" * 70)
    test_phase2_deck_and_action_contract()
    test_phase2_observation_layout_and_privacy()
    test_phase2_bench_energy_potion_and_masks()
    test_phase2_retreat_rules()
    test_phase2_weakness_knockouts_and_promotion()
    test_phase2_bench_out_deck_out_snapshot_and_alias_safety()
    print("=" * 70)
    print("              PHASE 2 ENVIRONMENT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_phase2_environment_tests()
