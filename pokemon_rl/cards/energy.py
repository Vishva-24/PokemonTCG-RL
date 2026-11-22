from pokemon_rl.cards.base import Card

class EnergyCard(Card):
    """A basic Energy card with a stable elemental type.

    ``Colorless`` keeps the Phase 1 generic-energy constructor usable.  Phase
    2 decks contain only Grass, Fire, Water, and Lightning basic Energy.
    """

    VALID_TYPES = {"Grass", "Fire", "Water", "Lightning", "Colorless"}

    def __init__(self, name: str = "Generic Energy", energy_type: str = "Colorless"):
        if energy_type not in self.VALID_TYPES:
            raise ValueError(f"Unsupported Energy type: {energy_type}")
        super().__init__(name, 'energy')
        self.energy_type = energy_type

    def __repr__(self) -> str:
        return f"EnergyCard(name='{self.name}', energy_type='{self.energy_type}')"
