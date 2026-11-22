from pokemon_rl.cards.pokemon import PokemonCard
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.player import Deck, Player

def test_deck_and_player():
    print("--- TESTING DECK AND PLAYER CLASSES ---")
    
    # 1. Create a dummy deck (similar to Phase 1 specifications: 6 cards for testing)
    cards = [
        PokemonCard("Tank", 100, 20, 1),
        PokemonCard("Attacker", 60, 40, 1),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        TrainerCard("Potion", "heal", 20)
    ]
    
    deck = Deck(cards)
    print(f"Created deck: {deck}")
    assert deck.remaining_count() == 6
    
    # Test shuffling
    deck.shuffle()
    print("Shuffled deck successfully.")
    
    # 2. Initialize Player
    player = Player("Red", deck)
    print(f"Initialized Player: {player}")
    
    # Test drawing cards
    drawn = player.draw_card()
    print(f"Drew card: {drawn}")
    assert len(player.hand) == 1
    assert player.deck.remaining_count() == 5
    
    # Draw until hand has 3 cards
    player.draw_card()
    player.draw_card()
    print(f"Player hand after drawing: {player.hand}")
    assert len(player.hand) == 3
    
    # 3. Test playing Pokémon to active spot
    pokemon_in_hand = None
    for card in player.hand:
        if isinstance(card, PokemonCard):
            pokemon_in_hand = card
            break
            
    if pokemon_in_hand:
        print(f"Attempting to play {pokemon_in_hand.name} to active spot...")
        player.play_pokemon_to_active(pokemon_in_hand)
        print(f"Player after playing active: {player}")
        assert player.active == pokemon_in_hand
        assert pokemon_in_hand not in player.hand
    else:
        print("No Pokemon card in hand to test playing to active.")
        
    # 4. Test state resetting
    print("Simulating damaging active and attaching energy...")
    if player.active:
        player.active.hp -= 30
        player.active.attached_energy += 1
        print(f"Active state before reset: {player.active}")
        
    print("Resetting player and deck...")
    player.reset()
    print(f"Player after reset: {player}")
    print(f"Deck after reset: {player.deck}")
    assert len(player.hand) == 0
    assert player.active is None
    assert player.deck.remaining_count() == 6
    
    # Verify that the active card itself has been reset inside the deck
    for c in player.deck.cards:
        if isinstance(c, PokemonCard) and c.name == "Tank":
            assert c.hp == c.max_hp
            assert c.attached_energy == 0
            
    print("Deck and Player tests passed successfully!")

if __name__ == "__main__":
    test_deck_and_player()