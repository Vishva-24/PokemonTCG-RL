class Card:
    """Base class for all cards in the Mini Pokémon TCG RL Environment."""
    def __init__(self, name: str, card_type: str):
        self.name = name
        self.card_type = card_type  # 'pokemon', 'energy', or 'trainer'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.card_type}')"