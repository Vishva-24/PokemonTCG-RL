"""Gymnasium wrapper and information-set encoding for Phase 2 Mini Pokémon TCG."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class gym:
        class Env:
            pass

    class spaces:
        class Box:
            def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

            def __repr__(self) -> str:
                return f"Box(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype})"

        class Discrete:
            def __init__(self, n: int):
                self.n = n

            def __repr__(self) -> str:
                return f"Discrete({self.n})"

from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.pokemon import Attack, PokemonCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.game import Action, GameEngine, GameState
from pokemon_rl.player import Deck


OBSERVATION_DIM = 50
ACTION_DIM = 70
ENERGY_TYPES = ("Grass", "Fire", "Water", "Lightning")
# This complete, one-to-one table is the locked Phase 2 observation contract.
# Keep the numeric literals below synchronized with these approved semantics.
OBSERVATION_INDEX = {
    0: "active_hp",
    1: "active_grass_energy",
    2: "active_fire_energy",
    3: "active_water_energy",
    4: "active_lightning_energy",
    5: "active_type_id",
    6: "bench1_hp",
    7: "bench1_grass_energy",
    8: "bench1_fire_energy",
    9: "bench1_water_energy",
    10: "bench1_lightning_energy",
    11: "bench1_type_id",
    12: "bench2_hp",
    13: "bench2_grass_energy",
    14: "bench2_fire_energy",
    15: "bench2_water_energy",
    16: "bench2_lightning_energy",
    17: "bench2_type_id",
    18: "hand_slot_0_type_id",
    19: "hand_slot_1_type_id",
    20: "hand_slot_2_type_id",
    21: "hand_slot_3_type_id",
    22: "hand_slot_4_type_id",
    23: "hand_slot_5_type_id",
    24: "hand_slot_6_type_id",
    25: "hand_slot_7_type_id",
    26: "deck_size",
    27: "energy_attached_this_turn",
    28: "retreated_this_turn",
    29: "forced_promotion_flag",
    30: "opponent_active_hp",
    31: "opponent_active_grass_energy",
    32: "opponent_active_fire_energy",
    33: "opponent_active_water_energy",
    34: "opponent_active_lightning_energy",
    35: "opponent_active_type_id",
    36: "opponent_bench1_hp",
    37: "opponent_bench1_grass_energy",
    38: "opponent_bench1_fire_energy",
    39: "opponent_bench1_water_energy",
    40: "opponent_bench1_lightning_energy",
    41: "opponent_bench1_type_id",
    42: "opponent_bench2_hp",
    43: "opponent_bench2_grass_energy",
    44: "opponent_bench2_fire_energy",
    45: "opponent_bench2_water_energy",
    46: "opponent_bench2_lightning_energy",
    47: "opponent_bench2_type_id",
    48: "opponent_hand_size",
    49: "opponent_deck_size",
}
if tuple(OBSERVATION_INDEX) != tuple(range(OBSERVATION_DIM)):
    raise RuntimeError("Phase 2 observation index table must contain each index 0 through 49 exactly once.")
POKEMON_TYPE_IDS = {"Grass": 1.0, "Fire": 2.0, "Water": 3.0, "Lightning": 4.0}
HAND_CARD_TYPE_IDS = {
    "empty": 0.0,
    "grass_pokemon": 1.0,
    "fire_pokemon": 2.0,
    "water_pokemon": 3.0,
    "lightning_pokemon": 4.0,
    "grass_energy": 5.0,
    "fire_energy": 6.0,
    "water_energy": 7.0,
    "lightning_energy": 8.0,
    "potion": 9.0,
    # Compatibility-only type for old callers creating EnergyCard() without
    # an element. Phase 2's supplied 20-card deck never uses this value.
    "colorless_energy": 10.0,
}


def _pokemon_type_id(pokemon: Optional[PokemonCard]) -> float:
    return POKEMON_TYPE_IDS.get(pokemon.pokemon_type, 0.0) if pokemon is not None else 0.0


def _hand_card_type_id(card: Any) -> float:
    """Encode only the viewing player's visible hand card identity/type."""
    if isinstance(card, PokemonCard):
        return _pokemon_type_id(card)
    if isinstance(card, EnergyCard):
        return HAND_CARD_TYPE_IDS.get(f"{card.energy_type.lower()}_energy", HAND_CARD_TYPE_IDS["colorless_energy"])
    if isinstance(card, TrainerCard) and card.effect_type == "heal":
        return HAND_CARD_TYPE_IDS["potion"]
    return HAND_CARD_TYPE_IDS["empty"]


def _write_pokemon_public_state(observation: np.ndarray, start: int, pokemon: Optional[PokemonCard]) -> None:
    """Write one exact six-field Active/Bench public Pokémon block."""
    if pokemon is None:
        return
    observation[start] = float(pokemon.hp)
    for offset, energy_type in enumerate(ENERGY_TYPES, start=1):
        observation[start + offset] = float(pokemon.energy_count(energy_type))
    observation[start + 5] = _pokemon_type_id(pokemon)


def get_phase2_information_set_observation(state: GameState, player_idx: int) -> np.ndarray:
    """Return the specified player's symmetric, private-information-safe view.

    The fixed indices are intentionally written here rather than constructed
    from variable blocks:

    | Indices | Meaning |
    | --- | --- |
    | 0 | Active HP |
    | 1-4 | Active Grass, Fire, Water, Lightning Energy |
    | 5 | Active Type ID |
    | 6 | Bench 1 HP |
    | 7-10 | Bench 1 Grass, Fire, Water, Lightning Energy |
    | 11 | Bench 1 Type ID |
    | 12 | Bench 2 HP |
    | 13-16 | Bench 2 Grass, Fire, Water, Lightning Energy |
    | 17 | Bench 2 Type ID |
    | 18-25 | Own hand slots 0-7 card type IDs |
    | 26 | Own deck size |
    | 27 | Energy attached this turn |
    | 28 | Retreated this turn |
    | 29 | Own forced-promotion flag |
    | 30 | Opponent Active HP |
    | 31-34 | Opponent Active Grass, Fire, Water, Lightning Energy |
    | 35 | Opponent Active Type ID |
    | 36 | Opponent Bench 1 HP |
    | 37-40 | Opponent Bench 1 Grass, Fire, Water, Lightning Energy |
    | 41 | Opponent Bench 1 Type ID |
    | 42 | Opponent Bench 2 HP |
    | 43-46 | Opponent Bench 2 Grass, Fire, Water, Lightning Energy |
    | 47 | Opponent Bench 2 Type ID |
    | 48 | Opponent hand size |
    | 49 | Opponent deck size |

    Board type IDs are 0 absent, 1 Grass, 2 Fire, 3 Water, 4 Lightning.
    Hand IDs are 0 empty; 1-4 Grass/Fire/Water/Lightning Pokémon; 5-8
    Grass/Fire/Water/Lightning Energy; 9 Potion; 10 Colorless Energy retained
    only for backwards-compatible ``EnergyCard()`` fixtures. Opponent hand and
    deck *contents* are never read or encoded.
    """
    if player_idx not in (0, 1):
        raise ValueError("player_idx must be 0 or 1.")

    observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    player = state.players[player_idx]
    opponent = state.players[1 - player_idx]

    _write_pokemon_public_state(observation, 0, player.active)
    _write_pokemon_public_state(observation, 6, player.bench[0])
    _write_pokemon_public_state(observation, 12, player.bench[1])
    for hand_slot, card in enumerate(player.hand[:8]):
        observation[18 + hand_slot] = _hand_card_type_id(card)
    observation[26] = float(player.deck.remaining_count())
    observation[27] = float(state.energy_attached_this_turn and state.current_player_idx == player_idx)
    observation[28] = float(state.retreated_this_turn and state.current_player_idx == player_idx)
    observation[29] = float(state.forced_promotion_player_idx == player_idx)

    _write_pokemon_public_state(observation, 30, opponent.active)
    _write_pokemon_public_state(observation, 36, opponent.bench[0])
    _write_pokemon_public_state(observation, 42, opponent.bench[1])
    observation[48] = float(len(opponent.hand))
    observation[49] = float(opponent.deck.remaining_count())
    return observation


class PokemonTCGEnv(gym.Env):
    """Phase 2 single-agent Red wrapper against an evaluation-only Random Blue.

    The wrapper owns no game rules. It converts the approved 70 actions to the
    core :class:`GameEngine` actions and exposes the fixed 50-dimensional Red
    information set. Action 3 delegates its unencoded retreat target to the
    engine's documented deterministic Bench 1 rule. Random Blue is used here
    only because this is the legacy single-agent environment; it is not part of
    future Deep CFR self-play.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.state: Optional[GameState] = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.state = GameEngine.initialize_game("Red", self._create_default_deck(), "Blue", self._create_default_deck())
        if self.state.current_player_idx == 1:
            self._run_opponent_turn()
        return self._get_observation(), {"turn_count": self.state.turn_count, "winner": None}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self.state.game_over:
            raise ValueError("Cannot call step on a finished game. Call reset() first.")
        if not isinstance(action_idx, (int, np.integer)) or not 0 <= int(action_idx) < ACTION_DIM:
            raise ValueError(f"Action index must be in [0, {ACTION_DIM - 1}].")
        mask = self.get_action_mask()
        if mask[int(action_idx)] != 1.0:
            raise ValueError(f"Action index {action_idx} is currently illegal in this state.")

        self.state = GameEngine.step(self.state, self._index_to_action(int(action_idx)))
        if not self.state.game_over and self.state.current_player_idx == 1:
            self._run_opponent_turn()

        terminated = self.state.game_over
        reward = 1.0 if terminated and self.state.winner_idx == 0 else -1.0 if terminated else 0.0
        return self._get_observation(), reward, terminated, False, {
            "turn_count": self.state.turn_count,
            "winner": "Red" if self.state.winner_idx == 0 else "Blue" if self.state.winner_idx == 1 else None,
        }

    def get_action_mask(self) -> np.ndarray:
        """Return the current Red action mask for the approved 70 indices."""
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        if self.state is None or self.state.game_over or self.state.current_player_idx != 0:
            return mask
        for action in GameEngine.get_legal_actions(self.state):
            action_idx = self._action_to_index(action)
            if action_idx is not None:
                mask[action_idx] = 1.0
        return mask

    @staticmethod
    def _index_to_action(action_idx: int) -> Action:
        """Decode exactly the approved 0-69 action mapping."""
        if action_idx == 0:
            return Action("end_turn")
        if action_idx == 1:
            return Action("attack", attack_idx=0)
        if action_idx == 2:
            return Action("attack", attack_idx=1)
        if action_idx == 3:
            return Action("retreat")
        if 4 <= action_idx <= 11:
            return Action("play_pokemon", card_idx=action_idx - 4, bench_slot=0)
        if 12 <= action_idx <= 19:
            return Action("play_pokemon", card_idx=action_idx - 12, bench_slot=1)
        if 20 <= action_idx <= 27:
            return Action("attach_energy", card_idx=action_idx - 20, target="active")
        if 28 <= action_idx <= 35:
            return Action("attach_energy", card_idx=action_idx - 28, bench_slot=0, target="bench")
        if 36 <= action_idx <= 43:
            return Action("attach_energy", card_idx=action_idx - 36, bench_slot=1, target="bench")
        if 44 <= action_idx <= 51:
            return Action("play_potion", card_idx=action_idx - 44, target="active")
        if 52 <= action_idx <= 59:
            return Action("play_potion", card_idx=action_idx - 52, bench_slot=0, target="bench")
        if 60 <= action_idx <= 67:
            return Action("play_potion", card_idx=action_idx - 60, bench_slot=1, target="bench")
        if action_idx == 68:
            return Action("promote", bench_slot=0)
        if action_idx == 69:
            return Action("promote", bench_slot=1)
        raise ValueError(f"Action index {action_idx} out of range [0, 69].")

    @staticmethod
    def _action_to_index(action: Action) -> Optional[int]:
        """Encode a concrete action, returning None only for unrepresentable input."""
        if action.action_type == "end_turn":
            return 0
        if action.action_type == "attack" and action.attack_idx in (0, 1):
            return 1 + action.attack_idx
        if action.action_type == "retreat":
            return 3
        if action.card_idx is not None and not 0 <= action.card_idx < 8:
            return None
        if action.action_type == "play_pokemon":
            return (4 if action.bench_slot == 0 else 12) + action.card_idx if action.bench_slot in (0, 1) else None
        if action.action_type == "attach_energy":
            if action.target == "active":
                return 20 + action.card_idx
            if action.target == "bench" and action.bench_slot in (0, 1):
                return (28 if action.bench_slot == 0 else 36) + action.card_idx
        if action.action_type == "play_potion":
            if action.target == "active":
                return 44 + action.card_idx
            if action.target == "bench" and action.bench_slot in (0, 1):
                return (52 if action.bench_slot == 0 else 60) + action.card_idx
        if action.action_type == "promote" and action.bench_slot in (0, 1):
            return 68 + action.bench_slot
        return None

    def _get_observation(self) -> np.ndarray:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        return get_phase2_information_set_observation(self.state, player_idx=0)

    @staticmethod
    def _create_default_deck() -> Deck:
        """Build the exact supplied 20-card Phase 2 deck."""
        return Deck([
            PokemonCard(
                "Pikachu", 70, card_id=948, expansion="ASC", collection_no=55,
                pokemon_type="Lightning", weakness="Fighting", retreat_cost=1,
                attacks=[
                    Attack("Tail Smack", {"Lightning": 1}, 10, "Lightning"),
                    Attack("Tiny Bolt", {"Lightning": 1, "Colorless": 1}, 30, "Lightning"),
                ],
            ),
            PokemonCard(
                "Charmander", 80, card_id=926, expansion="ASC", collection_no=20,
                pokemon_type="Fire", weakness="Water", retreat_cost=1,
                attacks=[Attack("Fire Claws", {"Fire": 1, "Colorless": 1}, 30, "Fire")],
            ),
            PokemonCard(
                "Swinub", 70, card_id=800, expansion="PFL", collection_no=23,
                pokemon_type="Water", weakness="Metal", retreat_cost=2,
                attacks=[
                    Attack("Stampede", {"Colorless": 1}, 10, "Water"),
                    Attack("Icy Snow", {"Water": 1, "Colorless": 1}, 20, "Water"),
                ],
            ),
            PokemonCard(
                "Applin", 40, card_id=149, expansion="SCR", collection_no=12,
                pokemon_type="Grass", weakness="Fire", retreat_cost=1,
                attacks=[Attack("Spray Fluid", {"Grass": 1}, 20, "Grass")],
            ),
            *[EnergyCard("Basic Grass Energy", "Grass") for _ in range(3)],
            *[EnergyCard("Basic Fire Energy", "Fire") for _ in range(3)],
            *[EnergyCard("Basic Water Energy", "Water") for _ in range(3)],
            *[EnergyCard("Basic Lightning Energy", "Lightning") for _ in range(3)],
            *[TrainerCard("Potion", "heal", 30, card_id=1117) for _ in range(4)],
        ])

    def _run_opponent_turn(self) -> None:
        """Run the existing Random Blue policy, including forced promotions."""
        while not self.state.game_over and self.state.current_player_idx == 1:
            legal_actions = GameEngine.get_legal_actions(self.state)
            if not legal_actions:
                raise RuntimeError("A non-terminal Blue state must have a legal action.")
            self.state = GameEngine.step(self.state, random.choice(legal_actions))
