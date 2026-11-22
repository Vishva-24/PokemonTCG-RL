from pokemon_rl.cards.base import Card

class TrainerCard(Card):
    """Class representing a Trainer card for utility actions."""
    def __init__(self, name: str, effect_type: str, heal_amount: int = 0, card_id: int | None = None):
        super().__init__(name, 'trainer')
        self.effect_type = effect_type  # e.g., 'heal'
        self.heal_amount = heal_amount
        self.card_id = card_id

    def __repr__(self) -> str:
        return f"TrainerCard(name='{self.name}', effect='{self.effect_type}', value={self.heal_amount})"
