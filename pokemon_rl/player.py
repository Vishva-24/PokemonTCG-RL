import random
from typing import List, Optional
from pokemon_rl.cards.base import Card
from pokemon_rl.cards.pokemon import PokemonCard

class Deck:
    """Manages a player's deck, including drawing, shuffling, and state resets."""
    def __init__(self, cards: List[Card]):
        self.initial_cards = list(cards)  # Retain original configuration for easy game resets
        self.cards = list(cards)

    def shuffle(self) -> None:
        """Shuffles the cards currently in the deck."""
        random.shuffle(self.cards)

    def draw(self) -> Optional[Card]:
        """Draws the top card of the deck. Returns None if empty."""
        if not self.cards:
            return None
        return self.cards.pop(0)

    def reset(self) -> None:
        """Restores the deck to its initial state, resetting any modified card properties."""
        self.cards = [c for c in self.initial_cards]
        for card in self.cards:
            if isinstance(card, PokemonCard):
                card.reset()

    def remaining_count(self) -> int:
        """Returns the number of cards remaining in the deck."""
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Deck(cards_left={self.remaining_count()})"


class Player:
    """Manages a player's Active spot, two fixed Bench slots, and card zones."""

    BENCH_CAPACITY = 2

    def __init__(self, name: str, deck: Deck):
        self.name = name
        self.deck = deck
        self.hand: List[Card] = []
        self.active: Optional[PokemonCard] = None
        # Fixed slots preserve the approved Bench 1 / Bench 2 action mapping.
        self.bench: List[Optional[PokemonCard]] = [None] * self.BENCH_CAPACITY
        self.trash: List[Card] = []

    def draw_card(self) -> Optional[Card]:
        """Draws a card from the deck and appends it to the hand."""
        card = self.deck.draw()
        if card:
            self.hand.append(card)
        return card

    def play_pokemon_to_active(self, pokemon_card: PokemonCard) -> None:
        """Plays a basic Pokémon card from hand directly to the Active spot."""
        if pokemon_card not in self.hand:
            raise ValueError(f"Card '{pokemon_card.name}' is not in hand.")
        if self.active is not None:
            raise ValueError("Active spot is already occupied.")
        
        self.hand.remove(pokemon_card)
        self.active = pokemon_card

    def play_pokemon_to_bench(self, pokemon_card: PokemonCard, bench_slot: int) -> None:
        """Move one Basic Pokémon from hand to a specified empty Bench slot."""
        if not 0 <= bench_slot < self.BENCH_CAPACITY:
            raise ValueError(f"Bench slot {bench_slot} is out of range.")
        if not isinstance(pokemon_card, PokemonCard):
            raise ValueError("Only Pokémon cards can be played to the Bench.")
        if pokemon_card not in self.hand:
            raise ValueError(f"Card '{pokemon_card.name}' is not in hand.")
        if self.bench[bench_slot] is not None:
            raise ValueError(f"Bench slot {bench_slot} is already occupied.")
        self.hand.remove(pokemon_card)
        self.bench[bench_slot] = pokemon_card

    def has_bench(self) -> bool:
        return any(card is not None for card in self.bench)

    def reset(self) -> None:
        """Resets the player's board, clearing the hand, active spot, and resetting the deck."""
        self.deck.reset()
        self.hand = []
        self.active = None
        self.bench = [None] * self.BENCH_CAPACITY
        self.trash = []

    def __repr__(self) -> str:
        active_name = self.active.name if self.active else "None"
        return (f"Player(name='{self.name}', hand_size={len(self.hand)}, active='{active_name}', "
                f"bench_size={sum(card is not None for card in self.bench)})")
