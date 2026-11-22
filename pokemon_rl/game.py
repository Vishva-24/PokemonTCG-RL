"""Phase 2 rules engine for the simplified Pokémon TCG environment.

The engine continues to own all game mutations.  It exposes concrete legal
``Action`` values while the Gym wrapper translates the approved fixed 70-action
space to and from those values.
"""

from __future__ import annotations

from typing import List, Optional

from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.pokemon import PokemonCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.player import Deck, Player


class Action:
    """A concrete legal game action.

    ``bench_slot`` is always zero-based internally: 0 is Bench 1 and 1 is
    Bench 2. The constructor keeps Phase 1 defaults for direct callers:
    Attack defaults to attack 1 and Energy/Potion default to the Active.
    """

    def __init__(self, action_type: str, card_idx: Optional[int] = None,
                 bench_slot: Optional[int] = None, attack_idx: Optional[int] = None,
                 target: Optional[str] = None):
        self.action_type = action_type
        self.card_idx = card_idx
        self.bench_slot = bench_slot
        self.attack_idx = 0 if action_type == "attack" and attack_idx is None else attack_idx
        self.target = "active" if action_type in {"attach_energy", "play_potion"} and target is None else target

    def __repr__(self) -> str:
        fields = [repr(self.action_type)]
        if self.card_idx is not None:
            fields.append(f"card_idx={self.card_idx}")
        if self.bench_slot is not None:
            fields.append(f"bench_slot={self.bench_slot}")
        if self.attack_idx is not None:
            fields.append(f"attack_idx={self.attack_idx}")
        if self.target is not None:
            fields.append(f"target={self.target!r}")
        return f"Action({', '.join(fields)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return False
        return (
            self.action_type == other.action_type
            and self.card_idx == other.card_idx
            and self.bench_slot == other.bench_slot
            and self.attack_idx == other.attack_idx
            and self.target == other.target
        )


class GameState:
    """Tracks all public and private state necessary for a Phase 2 match."""

    def __init__(self, players: List[Player]):
        if len(players) != 2:
            raise ValueError("A game requires exactly two players.")
        self.players = players
        self.current_player_idx = 0
        self.energy_attached_this_turn = False
        self.retreated_this_turn = False
        # The defeated player who must select a Bench card, or None.
        self.forced_promotion_player_idx: Optional[int] = None
        # Number of opposing Active Pokémon each player has knocked out.
        self.knockout_counts = [0, 0]
        self.game_over = False
        self.winner_idx: Optional[int] = None
        self.turn_count = 1

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    @property
    def opponent(self) -> Player:
        return self.players[1 - self.current_player_idx]

    @property
    def forced_promotion_pending(self) -> bool:
        return self.forced_promotion_player_idx is not None

    def switch_turn(self) -> None:
        """Switch active player and reset per-turn attachment/retreat limits."""
        self.current_player_idx = 1 - self.current_player_idx
        self.energy_attached_this_turn = False
        self.retreated_this_turn = False
        self.turn_count += 1

    def __repr__(self) -> str:
        p1, p2 = self.players
        active_turn = self.current_player.name
        return (
            f"GameState(Turn={self.turn_count}, ActivePlayer='{active_turn}', "
            f"P1={p1.name}[Active={p1.active.name if p1.active else 'None'}, "
            f"Bench={sum(card is not None for card in p1.bench)}, Hand={len(p1.hand)}], "
            f"P2={p2.name}[Active={p2.active.name if p2.active else 'None'}, "
            f"Bench={sum(card is not None for card in p2.bench)}, Hand={len(p2.hand)}], "
            f"KOs={self.knockout_counts}, ForcedPromotion={self.forced_promotion_player_idx}, "
            f"GameOver={self.game_over})"
        )


class GameEngine:
    """Enforces the approved Phase 2 game state machine.

    The one target not encoded in the fixed action space is Retreat. When two
    Bench slots are occupied, Retreat deterministically selects Bench 1 (slot
    0). Retreat payment also has a deterministic order: the most recently
    attached Energy cards are moved to that player's Trash first.
    """

    VISIBLE_HAND_SLOTS = 8

    @staticmethod
    def initialize_game(p1_name: str, p1_deck: Deck, p2_name: str, p2_deck: Deck) -> GameState:
        """Set one Basic Active Pokémon each, then draw Phase 1's opening hand."""
        p1 = Player(p1_name, p1_deck)
        p2 = Player(p2_name, p2_deck)
        p1.reset()
        p2.reset()

        for player in (p1, p2):
            pokemon_card: Optional[PokemonCard] = None
            non_pokemon = []
            while player.deck.remaining_count() > 0:
                card = player.deck.draw()
                if isinstance(card, PokemonCard):
                    pokemon_card = card
                    break
                if card is not None:
                    non_pokemon.append(card)
            if pokemon_card is None:
                raise ValueError(f"Player {player.name}'s deck does not contain any Pokémon cards!")
            player.active = pokemon_card
            player.deck.cards.extend(non_pokemon)
            player.deck.shuffle()

        for player in (p1, p2):
            for _ in range(3):
                if player.draw_card() is None:
                    raise ValueError("Deck ran out of cards during opening setup.")

        state = GameState([p1, p2])
        if state.current_player.draw_card() is None:
            raise ValueError("Starting player's deck ran out of cards during opening setup.")
        return state

    @classmethod
    def get_legal_actions(cls, state: GameState) -> List[Action]:
        """Return exact current-state actions representable by the 70-action API."""
        if state.game_over:
            return []

        player = state.current_player
        if state.forced_promotion_pending:
            if state.forced_promotion_player_idx != state.current_player_idx:
                raise RuntimeError("Forced-promotion player must be the current player.")
            return [
                Action("promote", bench_slot=slot)
                for slot, pokemon in enumerate(player.bench)
                if pokemon is not None
            ]

        actions: List[Action] = [Action("end_turn")]
        if player.active is not None:
            for attack_idx in range(len(player.active.attacks)):
                if player.active.can_use_attack(attack_idx):
                    actions.append(Action("attack", attack_idx=attack_idx))
            if cls._can_retreat(state):
                actions.append(Action("retreat"))

        # Hand actions are deliberately capped at slots 0--7. The fixed
        # environment action space cannot represent later slots, so treating a
        # later slot as legal would create an aliasing bug.
        for card_idx, card in enumerate(player.hand[:cls.VISIBLE_HAND_SLOTS]):
            if isinstance(card, PokemonCard):
                for bench_slot, bench_card in enumerate(player.bench):
                    if bench_card is None:
                        actions.append(Action("play_pokemon", card_idx=card_idx, bench_slot=bench_slot))
            elif isinstance(card, EnergyCard) and not state.energy_attached_this_turn:
                if player.active is not None:
                    actions.append(Action("attach_energy", card_idx=card_idx, target="active"))
                for bench_slot, bench_card in enumerate(player.bench):
                    if bench_card is not None:
                        actions.append(Action("attach_energy", card_idx=card_idx, bench_slot=bench_slot, target="bench"))
            elif isinstance(card, TrainerCard) and card.effect_type == "heal":
                if player.active is not None and player.active.hp < player.active.max_hp:
                    actions.append(Action("play_potion", card_idx=card_idx, target="active"))
                for bench_slot, bench_card in enumerate(player.bench):
                    if bench_card is not None and bench_card.hp < bench_card.max_hp:
                        actions.append(Action("play_potion", card_idx=card_idx, bench_slot=bench_slot, target="bench"))
        return actions

    @classmethod
    def step(cls, state: GameState, action: Action) -> GameState:
        """Validate then apply one concrete Action.

        Validation against ``get_legal_actions`` immediately before mutation
        prevents stale hand indices or action aliases from becoming a different
        card play.
        """
        if state.game_over:
            raise ValueError("Cannot perform actions. The game is already over.")
        if action not in cls.get_legal_actions(state):
            raise ValueError(f"Action is illegal in the current state: {action!r}")

        player = state.current_player
        opponent = state.opponent

        if action.action_type == "end_turn":
            cls._end_turn_sequence(state)
        elif action.action_type == "play_pokemon":
            assert action.card_idx is not None and action.bench_slot is not None
            card = player.hand[action.card_idx]
            if not isinstance(card, PokemonCard):
                raise RuntimeError("Legal-action validation accepted a non-Pokémon Bench play.")
            player.play_pokemon_to_bench(card, action.bench_slot)
        elif action.action_type == "attach_energy":
            assert action.card_idx is not None
            card = player.hand[action.card_idx]
            if not isinstance(card, EnergyCard):
                raise RuntimeError("Legal-action validation accepted a non-Energy attachment.")
            target = cls._target_pokemon(player, action)
            player.hand.pop(action.card_idx)
            target.attach_energy(card)
            state.energy_attached_this_turn = True
        elif action.action_type == "play_potion":
            assert action.card_idx is not None
            card = player.hand[action.card_idx]
            if not isinstance(card, TrainerCard) or card.effect_type != "heal":
                raise RuntimeError("Legal-action validation accepted a non-healing Trainer.")
            target = cls._target_pokemon(player, action)
            player.hand.pop(action.card_idx)
            target.hp = min(target.max_hp, target.hp + card.heal_amount)
            player.trash.append(card)
        elif action.action_type == "retreat":
            assert player.active is not None
            bench_slot = cls._retreat_target_slot(player)
            paid_energy = player.active.remove_retreat_energy(player.active.retreat_cost)
            player.trash.extend(paid_energy)
            player.active, player.bench[bench_slot] = player.bench[bench_slot], player.active
            state.retreated_this_turn = True
        elif action.action_type == "attack":
            assert player.active is not None and action.attack_idx is not None
            attack = player.active.attacks[action.attack_idx]
            assert opponent.active is not None
            damage = attack.damage
            if attack.attack_type == opponent.active.weakness:
                damage *= 2
            opponent.active.hp = max(0, opponent.active.hp - damage)
            if opponent.active.hp == 0:
                cls._handle_knockout(state, attacker_idx=state.current_player_idx)
            else:
                cls._end_turn_sequence(state)
        elif action.action_type == "promote":
            assert action.bench_slot is not None
            player.active = player.bench[action.bench_slot]
            player.bench[action.bench_slot] = None
            state.forced_promotion_player_idx = None
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
        return state

    @staticmethod
    def _target_pokemon(player: Player, action: Action) -> PokemonCard:
        if action.target == "active":
            if player.active is None:
                raise RuntimeError("Active target disappeared after legal-action validation.")
            return player.active
        if action.target == "bench" and action.bench_slot is not None:
            target = player.bench[action.bench_slot]
            if target is None:
                raise RuntimeError("Bench target disappeared after legal-action validation.")
            return target
        raise ValueError(f"Invalid action target: {action!r}")

    @staticmethod
    def _retreat_target_slot(player: Player) -> int:
        for slot, pokemon in enumerate(player.bench):
            if pokemon is not None:
                return slot
        raise ValueError("Cannot retreat without a Bench Pokémon.")

    @classmethod
    def _can_retreat(cls, state: GameState) -> bool:
        active = state.current_player.active
        return (
            active is not None
            and not state.retreated_this_turn
            and state.current_player.has_bench()
            and active.attached_energy >= active.retreat_cost
        )

    @classmethod
    def _handle_knockout(cls, state: GameState, attacker_idx: int) -> None:
        """Trash the defeated Active and resolve two-KO/bench-out/promotion."""
        defender_idx = 1 - attacker_idx
        defender = state.players[defender_idx]
        if defender.active is None:
            raise RuntimeError("Cannot knock out a player without an Active Pokémon.")
        defender.trash.extend(defender.active.attached_energy_cards)
        defender.active.attached_energy_cards = []
        defender.trash.append(defender.active)
        defender.active = None
        state.knockout_counts[attacker_idx] += 1

        if state.knockout_counts[attacker_idx] >= 2 or not defender.has_bench():
            state.game_over = True
            state.winner_idx = attacker_idx
            return

        # Preserve the Phase 1 convention that attacking ends the turn. The
        # defeated player draws for the new turn and must then choose a Bench
        # promotion before any normal action is available.
        cls._end_turn_sequence(state)
        if not state.game_over:
            state.forced_promotion_player_idx = defender_idx

    @staticmethod
    def _end_turn_sequence(state: GameState) -> None:
        """Switch turn, reset per-turn state, draw, and resolve deck-out."""
        state.switch_turn()
        drawn = state.current_player.draw_card()
        if drawn is None:
            state.game_over = True
            state.winner_idx = 1 - state.current_player_idx
