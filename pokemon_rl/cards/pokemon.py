from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pokemon_rl.cards.base import Card
from pokemon_rl.cards.energy import EnergyCard


@dataclass(frozen=True)
class Attack:
    """One simplified Phase 2 attack and its exact basic-Energy cost."""

    name: str
    energy_cost: Dict[str, int]
    damage: int
    attack_type: str

class PokemonCard(Card):
    """A Basic Pokémon with typed attached Energy and one or two attacks.

    The original positional constructor remains valid for Phase 1 callers.  It
    creates one Colorless attack from ``attack_damage`` and ``energy_required``.
    Phase 2 supplies exact metadata via keyword-only fields.
    """

    def __init__(self, name: str, hp: int, attack_damage: int = 0, energy_required: int = 0,
                 *, card_id: Optional[int] = None, expansion: Optional[str] = None,
                 collection_no: Optional[int] = None, pokemon_type: str = "Colorless",
                 weakness: Optional[str] = None, retreat_cost: int = 0,
                 attacks: Optional[List[Attack]] = None):
        super().__init__(name, 'pokemon')
        if hp <= 0:
            raise ValueError("Pokemon HP must be positive.")
        if retreat_cost < 0:
            raise ValueError("Retreat cost cannot be negative.")
        self.max_hp = hp
        self.hp = hp  # Tracks remaining HP during battles
        self.card_id = card_id
        self.expansion = expansion
        self.collection_no = collection_no
        self.pokemon_type = pokemon_type
        self.weakness = weakness
        self.retreat_cost = retreat_cost
        self.attacks = list(attacks) if attacks is not None else [
            Attack("Attack", {"Colorless": energy_required}, attack_damage, pokemon_type)
        ]
        if not self.attacks:
            raise ValueError("A Pokemon must have at least one attack.")
        # Keep individual cards so Energy colours can be observed, validated,
        # and moved to the Trash when retreat costs are paid.
        self.attached_energy_cards: List[EnergyCard] = []

    @property
    def attack_damage(self) -> int:
        """Compatibility alias for the original single-attack Phase 1 API."""
        return self.attacks[0].damage

    @property
    def energy_required(self) -> int:
        """Compatibility alias for the original first-attack total cost."""
        return sum(self.attacks[0].energy_cost.values())

    @property
    def attached_energy(self) -> int:
        """Total attached Energy, retained for legacy callers."""
        return len(self.attached_energy_cards)

    @attached_energy.setter
    def attached_energy(self, count: int) -> None:
        """Compatibility setter used only by older tests and fixtures.

        It creates Colorless cards, which can pay Colorless costs but not a
        Phase 2 elemental requirement.  Normal gameplay always attaches a
        concrete :class:`EnergyCard` through ``attach_energy``.
        """
        if count < 0:
            raise ValueError("Attached Energy cannot be negative.")
        self.attached_energy_cards = [EnergyCard() for _ in range(int(count))]

    def attach_energy(self, energy: EnergyCard) -> None:
        if not isinstance(energy, EnergyCard):
            raise ValueError("Only EnergyCard instances can be attached.")
        self.attached_energy_cards.append(energy)

    def energy_count(self, energy_type: str) -> int:
        return sum(card.energy_type == energy_type for card in self.attached_energy_cards)

    def can_use_attack(self, attack_idx: int) -> bool:
        if not 0 <= attack_idx < len(self.attacks):
            return False
        cost = self.attacks[attack_idx].energy_cost
        # Colour requirements are paid by matching basic Energy only.
        for energy_type, required in cost.items():
            if energy_type != "Colorless" and self.energy_count(energy_type) < required:
                return False
        colourless_required = cost.get("Colorless", 0)
        typed_required = sum(required for energy_type, required in cost.items() if energy_type != "Colorless")
        return self.attached_energy >= typed_required + colourless_required

    def remove_retreat_energy(self, count: int) -> List[EnergyCard]:
        """Remove the most recently attached cards, deterministically."""
        if count < 0 or self.attached_energy < count:
            raise ValueError("Not enough attached Energy to pay retreat cost.")
        removed = self.attached_energy_cards[-count:] if count else []
        if count:
            del self.attached_energy_cards[-count:]
        return removed

    def reset(self):
        """Resets dynamic battle state when shuffling back into the deck."""
        self.hp = self.max_hp
        self.attached_energy_cards = []

    def __repr__(self) -> str:
        return (f"PokemonCard(name='{self.name}', HP={self.hp}/{self.max_hp}, "
                f"Damage={self.attack_damage}, EnergyReq={self.energy_required}, "
                f"EnergyAttached={self.attached_energy}, Type={self.pokemon_type})")
