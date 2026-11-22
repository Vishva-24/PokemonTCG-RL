import random
from pokemon_rl.cards.pokemon import PokemonCard
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.player import Deck
from pokemon_rl.game import GameEngine, Action, GameState

def make_test_deck() -> Deck:
    """Creates a standardized 20-card deck for testing to prevent premature deck-out."""
    return Deck([
        PokemonCard("Tank", 100, 20, 1),
        PokemonCard("Attacker", 60, 40, 1),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        EnergyCard(),
        TrainerCard("Potion", "heal", 20),
        TrainerCard("Potion", "heal", 20),
        TrainerCard("Potion", "heal", 20)
    ])

def test_game_engine_deterministic():
    print("--- RUNNING DETERMINISTIC FUNCTIONAL TEST ---")
    
    deck1 = make_test_deck()
    deck2 = make_test_deck()
    
    # Store initial deck count before starting
    initial_deck_size = deck1.remaining_count()
    
    # 1. Initialize game
    state = GameEngine.initialize_game("Red", deck1, "Blue", deck2)
    print(f"Game initialized: {state}")
    
    assert len(state.players) == 2
    assert state.current_player_idx == 0
    assert state.players[0].active is not None
    assert state.players[1].active is not None
    
    # Starting hand is 3 cards. Red drew 1 more for turn start -> 4 cards in hand.
    assert len(state.players[0].hand) == 4
    assert len(state.players[1].hand) == 3
    
    # Verify deck counts dynamically (Red draws 5 total during setup/turn start, Blue draws 4 total)
    assert state.players[0].deck.remaining_count() == initial_deck_size - 5
    assert state.players[1].deck.remaining_count() == initial_deck_size - 4
    
    # 2. Check legal actions
    legal_actions = GameEngine.get_legal_actions(state)
    print(f"Legal actions for Turn 1 Red: {legal_actions}")
    assert any(act.action_type == "end_turn" for act in legal_actions)
    
    # Let's find an energy card in Red's hand to play
    energy_idx = next((idx for idx, c in enumerate(state.players[0].hand) if isinstance(c, EnergyCard)), None)
    if energy_idx is not None:
        print(f"Red playing EnergyCard from hand at index {energy_idx}...")
        attach_action = Action("attach_energy", card_idx=energy_idx)
        state = GameEngine.step(state, attach_action)
        print(f"After Energy Attachment: {state}")
        assert state.players[0].active.attached_energy == 1
        assert state.energy_attached_this_turn is True
        
        # Test that attaching energy again in the same turn is not legal
        post_attach_actions = GameEngine.get_legal_actions(state)
        assert not any(act.action_type == "attach_energy" for act in post_attach_actions)
        
        # Red should now be able to attack since Tank requires 1 energy
        assert any(act.action_type == "attack" for act in post_attach_actions)
        print("Red attacking Blue!")
        opponent_hp_before = state.players[1].active.hp
        attack_damage = state.players[0].active.attack_damage
        
        # Step attack
        state = GameEngine.step(state, Action("attack"))
        print(f"After attack, turn passes. Current state: {state}")
        assert state.players[1].active.hp == opponent_hp_before - attack_damage
        assert state.current_player_idx == 1
        assert len(state.players[1].hand) == 4
    else:
        print("No Energy card in starting hand. Ending turn.")
        state = GameEngine.step(state, Action("end_turn"))
        assert state.current_player_idx == 1

    # Simulate battle until knockout with prioritized actions (Attack > Attach > Potion > End Turn)
    print("\n--- SIMULATING DETERMINISTIC BATTLE UNTIL KNOCKOUT ---")
    while not state.game_over and state.turn_count < 100:
        actions = GameEngine.get_legal_actions(state)
        attack_act = next((act for act in actions if act.action_type == "attack"), None)
        attach_act = next((act for act in actions if act.action_type == "attach_energy"), None)
        potion_act = next((act for act in actions if act.action_type == "play_potion"), None)
        
        if attack_act:
            print(f"{state.current_player.name} attacks!")
            state = GameEngine.step(state, attack_act)
        elif attach_act:
            print(f"{state.current_player.name} attaches energy.")
            state = GameEngine.step(state, attach_act)
        elif potion_act:
            print(f"{state.current_player.name} plays Potion!")
            state = GameEngine.step(state, potion_act)
        else:
            print(f"{state.current_player.name} ends turn.")
            state = GameEngine.step(state, Action("end_turn"))
            
    print(f"Deterministic game finished! State: {state}")
    assert state.game_over is True
    assert state.winner_idx is not None
    winner_name = state.players[state.winner_idx].name
    print(f"Winner: {winner_name}\n")


# def test_game_fairness_randomized():
#     print("--- RUNNING FAIRNESS/RANDOMNESS TEST (100 GAMES) ---")
    
#     red_wins = 0
#     blue_wins = 0
#     total_games = 100
    
#     # Set a seed for reproducibility
#     random.seed(42)
    
#     for game_idx in range(total_games):
#         deck1 = make_test_deck()
#         deck2 = make_test_deck()
        
#         # Randomize who starts by swapping player names and decks.
#         # GameEngine always makes player index 0 (the first argument) go first.
#         # Swapping names/decks randomizes the starting player without editing GameEngine rules.
#         swap_starts = random.choice([True, False])
        
#         if swap_starts:
#             # Blue goes first (assigned to players[0]), Red goes second (assigned to players[1])
#             state = GameEngine.initialize_game("Blue", deck2, "Red", deck1)
#         else:
#             # Red goes first (assigned to players[0]), Blue goes second (assigned to players[1])
#             state = GameEngine.initialize_game("Red", deck1, "Blue", deck2)
            
#         while not state.game_over:
#             legal_actions = GameEngine.get_legal_actions(state)
#             if not legal_actions:
#                 state = GameEngine.step(state, Action("end_turn"))
#                 continue
                
#             # RandomAgent policy: choose randomly from currently legal actions
#             chosen_action = random.choice(legal_actions)
#             state = GameEngine.step(state, chosen_action)
            
#         winner_name = state.players[state.winner_idx].name
#         if winner_name == "Red":
#             red_wins += 1
#         elif winner_name == "Blue":
#             blue_wins += 1
            
#     print(f"Red wins: {red_wins}")
#     print(f"Blue wins: {blue_wins}")
#     print(f"Total games: {total_games}")
    
#     # Assertions to verify the fairness test:
#     assert red_wins + blue_wins == total_games
#     assert red_wins > 0, "Red must win at least one game under random choice"
#     assert blue_wins > 0, "Blue must win at least one game under random choice"
#     print("Fairness/randomness test completed and passed successfully!")


if __name__ == "__main__":
    test_game_engine_deterministic()
    # test_game_fairness_randomized()