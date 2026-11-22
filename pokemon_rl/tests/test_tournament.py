"""Tests for the isolated four-algorithm tournament evaluator."""

from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pokemon_rl.evaluation.tournament import (
    CROSS_MATCHUPS,
    IllegalActionError,
    OrientationPlan,
    TournamentConfig,
    _action_mask,
    _game_rngs,
    _orientation_summary,
    _pairwise_matrix,
    _seeded_initial_state,
    _terminal_outcome,
    build_orientation_plans,
    run_single_game,
    run_tournament,
    wilson_interval,
)
from pokemon_rl.evaluation.tournament_agents import (
    ALGORITHM_NAMES,
    DEFAULT_CHECKPOINTS,
    DeepCFRStrategyAgent,
    DQNAgent,
    LSTMPPOAgent,
    PPOAgent,
    default_checkpoint_path,
    load_agent,
)
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv, get_phase2_information_set_observation
from pokemon_rl.game import GameEngine


DEVICE = torch.device("cpu")


def _load_all_agents():
    return {
        "DQN": (load_agent("DQN", 0, DEVICE), load_agent("DQN", 1, DEVICE)),
        "PPO": (load_agent("PPO", 0, DEVICE), load_agent("PPO", 1, DEVICE)),
        "LSTM-PPO": (load_agent("LSTM-PPO", 0, DEVICE), load_agent("LSTM-PPO", 1, DEVICE)),
        "Deep CFR": (load_agent("Deep CFR", 0, DEVICE), load_agent("Deep CFR", 1, DEVICE)),
    }


class TournamentTests(unittest.TestCase):
    def test_checkpoint_inventory_and_intended_adapters_load(self):
        agents = _load_all_agents()
        assert Path(default_checkpoint_path("DQN", 0)).name == "final_model.pt"
        assert Path(default_checkpoint_path("PPO", 0)).name == "final_ppo_model.pt"
        assert Path(default_checkpoint_path("LSTM-PPO", 1)).name == "final_lstm_ppo_actor.pt"
        assert Path(default_checkpoint_path("Deep CFR", 0)).name == DEFAULT_CHECKPOINTS["Deep CFR"]["red"].split("/")[-1]
        assert Path(default_checkpoint_path("Deep CFR", 1)).name == DEFAULT_CHECKPOINTS["Deep CFR"]["blue"].split("/")[-1]
        assert isinstance(agents["DQN"][0], DQNAgent)
        assert isinstance(agents["PPO"][0], PPOAgent)
        assert isinstance(agents["LSTM-PPO"][0], LSTMPPOAgent)
        assert isinstance(agents["Deep CFR"][0], DeepCFRStrategyAgent)
        for algorithm, pair in agents.items():
            assert pair[0] is not pair[1]
            assert pair[0].algorithm == algorithm
            assert pair[1].algorithm == algorithm



    def test_all_agents_accept_50d_view_and_select_only_legal_actions(self):
        agents = _load_all_agents()
        state = _seeded_initial_state(108)
        red_observation = get_phase2_information_set_observation(state, 0)
        red_mask = _action_mask(state, 0)
        assert red_observation.shape == (OBSERVATION_DIM,)
        assert red_mask.shape == (ACTION_DIM,)
        for algorithm, pair in agents.items():
            action = pair[0].select_action(red_observation, red_mask, np.random.default_rng(44))
            assert 0 <= action < ACTION_DIM
            assert red_mask[action] == 1.0, algorithm

        state = GameEngine.step(state, PokemonTCGEnv._index_to_action(0))
        blue_observation = get_phase2_information_set_observation(state, 1)
        blue_mask = _action_mask(state, 1)
        for algorithm, pair in agents.items():
            action = pair[1].select_action(blue_observation, blue_mask, np.random.default_rng(77))
            assert 0 <= action < ACTION_DIM
            assert blue_mask[action] == 1.0, algorithm



    def test_forced_promotion_masks_are_respected_by_every_adapter(self):
        agents = _load_all_agents()
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        masks = []
        one_slot = np.zeros(ACTION_DIM, dtype=np.float32)
        one_slot[68] = 1.0
        masks.append((one_slot, 68))
        two_slots = np.zeros(ACTION_DIM, dtype=np.float32)
        two_slots[[68, 69]] = 1.0
        masks.append((two_slots, None))
        for pair in agents.values():
            for agent in pair:
                for mask, only_action in masks:
                    action = agent.select_action(observation, mask, np.random.default_rng(998))
                    assert mask[action] == 1.0
                    if only_action is not None:
                        assert action == only_action



    def test_observations_are_player_relative_and_hide_other_hand_contents(self):
        state = _seeded_initial_state(12)
        state.players[1].hand = [
            PokemonTCGEnv._create_default_deck().cards[4],
            PokemonTCGEnv._create_default_deck().cards[7],
            PokemonTCGEnv._create_default_deck().cards[10],
        ]
        red_before = get_phase2_information_set_observation(state, 0)
        blue_before = get_phase2_information_set_observation(state, 1)
        red_mask_before = _action_mask(state, 0)
        state.players[1].hand[0], state.players[1].hand[1] = state.players[1].hand[1], state.players[1].hand[0]
        red_after = get_phase2_information_set_observation(state, 0)
        blue_after = get_phase2_information_set_observation(state, 1)
        assert np.array_equal(red_before, red_after)
        assert not np.array_equal(blue_before, blue_after)
        assert red_before[48] == len(state.players[1].hand)
        assert np.array_equal(red_mask_before, _action_mask(state, 0))



    def test_lstm_hidden_states_are_independent_and_reset_per_game(self):
        red = load_agent("LSTM-PPO", 0, DEVICE)
        blue = load_agent("LSTM-PPO", 1, DEVICE)
        assert isinstance(red, LSTMPPOAgent) and isinstance(blue, LSTMPPOAgent)
        assert red is not blue
        red_before = tuple(t.clone() for t in red.hidden_state)
        blue_before = tuple(t.clone() for t in blue.hidden_state)
        state = _seeded_initial_state(99)
        observation = get_phase2_information_set_observation(state, 0)
        mask = _action_mask(state, 0)
        red.select_action(observation, mask, np.random.default_rng(1))
        assert any(not torch.equal(before, after) for before, after in zip(red_before, red.hidden_state))
        assert all(torch.equal(before, after) for before, after in zip(blue_before, blue.hidden_state))
        red.reset()
        assert all(torch.count_nonzero(value) == 0 for value in red.hidden_state)



    def test_same_checkpoint_copies_have_independent_model_parameters(self):
        first = DQNAgent.from_checkpoint(default_checkpoint_path("DQN", 0), DEVICE)
        second = DQNAgent.from_checkpoint(default_checkpoint_path("DQN", 1), DEVICE)
        assert first is not second
        assert first.model.fc1.weight.data_ptr() != second.model.fc1.weight.data_ptr()
        second_before = second.model.fc1.weight.detach().clone()
        with torch.no_grad():
            first.model.fc1.weight.add_(1.0)
        assert torch.equal(second.model.fc1.weight, second_before)



    def test_deep_cfr_selects_role_specific_strategy_checkpoints(self):
        red = load_agent("Deep CFR", 0, DEVICE)
        blue = load_agent("Deep CFR", 1, DEVICE)
        assert isinstance(red, DeepCFRStrategyAgent) and isinstance(blue, DeepCFRStrategyAgent)
        assert red.player_idx == 0 and blue.player_idx == 1
        assert red.checkpoint_path.name == "final_deep_cfr_red_strategy_masked_ce_iteration_weighted.pt"
        assert blue.checkpoint_path.name == "final_deep_cfr_blue_strategy_masked_ce_iteration_weighted.pt"
        assert red.strategy_network is not blue.strategy_network



    def test_game_rng_is_independent_and_policy_streams_are_separate(self):
        before = random.getstate()
        first = _seeded_initial_state(123)
        after = random.getstate()
        second = _seeded_initial_state(123)
        assert before == after
        assert first.players[0].active.name == second.players[0].active.name
        assert [card.name for card in first.players[0].deck.cards] == [card.name for card in second.players[0].deck.cards]
        plan = OrientationPlan("cross_play", "DQN vs Deep CFR", "DQN_red__DeepCFR_blue", "DQN", "Deep CFR", 1, 2)
        red_rng, blue_rng = _game_rngs(TournamentConfig(seed=10), plan, 10)
        assert red_rng is not blue_rng
        assert red_rng.integers(0, 2**32) != blue_rng.integers(0, 2**32)



    def test_terminal_reasons_use_existing_game_state_information(self):
        state = _seeded_initial_state(5)
        state.game_over = True
        state.winner_idx = 0
        state.knockout_counts[0] = 2
        assert _terminal_outcome(state) == ("two_knockout_victory", None)

        state.knockout_counts[0] = 1
        state.players[1].active = None
        state.players[1].bench = [None, None]
        assert _terminal_outcome(state) == ("bench_out", 1)

        state.players[1].active = state.players[0].active
        state.players[1].bench = [None, None]
        state.current_player_idx = 1
        state.players[1].deck.cards = []
        assert _terminal_outcome(state) == ("deck_out", 1)


    class EndTurnAgent:
        algorithm = "test end-turn"

        def reset(self) -> None:
            return None

        def select_action(self, observation, action_mask, rng):
            assert observation.shape == (OBSERVATION_DIM,)
            assert action_mask.shape == (ACTION_DIM,)
            if action_mask[0] != 1.0:
                raise AssertionError("The no-attack smoke state should allow End Turn.")
            return 0



    def test_game_terminates_by_deckout_and_cap_is_not_a_loss(self):
        plan = OrientationPlan("self_play", "DQN vs DQN", "one", "DQN", "DQN", 1, 2)
        terminal = run_single_game(
            plan, 741, 200, self.EndTurnAgent(), self.EndTurnAgent(), np.random.default_rng(1), np.random.default_rng(2)
        )
        assert terminal["terminal"] is True
        assert terminal["terminal_reason"] == "deck_out"
        assert terminal["winner"] in {"Red", "Blue"}
        assert terminal["decisions"] < 200
        capped = run_single_game(
            plan, 741, 1, self.EndTurnAgent(), self.EndTurnAgent(), np.random.default_rng(1), np.random.default_rng(2)
        )
        assert capped["terminal"] is False
        assert capped["step_cap_reached"] is True
        assert capped["winner"] == ""
        assert capped["red_reward"] == capped["blue_reward"] == 0.0


    def test_illegal_adapter_action_is_rejected_before_engine_step(self):
        class IllegalAgent:
            algorithm = "test illegal"

            def reset(self) -> None:
                return None

            def select_action(self, observation, action_mask, rng):
                assert action_mask[69] == 0.0
                return 69

        plan = OrientationPlan("self_play", "DQN vs DQN", "one", "DQN", "DQN", 1, 2)
        agent = IllegalAgent()
        with patch.object(GameEngine, "step") as engine_step:
            with self.assertRaises(IllegalActionError):
                run_single_game(
                    plan, 742, 200, agent, self.EndTurnAgent(),
                    np.random.default_rng(1), np.random.default_rng(2),
                )
            engine_step.assert_not_called()



    def test_schedule_counts_and_orientation_coverage(self):
        assert len(CROSS_MATCHUPS) == 6
        assert TournamentConfig().expected_games == 4000
        assert len(build_orientation_plans("cross_play_only")) == 12
        assert len(build_orientation_plans("self_play_only")) == 8
        assert len(build_orientation_plans("all")) == 20
        cross = build_orientation_plans("cross_play_only")
        assert sum(plan.algorithm_red == "DQN" and plan.algorithm_blue == "PPO" for plan in cross) == 1
        assert sum(plan.algorithm_red == "PPO" and plan.algorithm_blue == "DQN" for plan in cross) == 1



    def test_wilson_interval_and_orientation_metrics(self):
        low, high = wilson_interval(50, 100)
        assert 40.0 < low < 40.5
        assert 59.5 < high < 60.0
        records = [
            {"match_type": "cross_play", "matchup": "DQN vs PPO", "orientation": "DQN_red", "algorithm_red": "DQN", "algorithm_blue": "PPO", "winner": "Red", "terminal": True, "step_cap_reached": False, "terminal_reason": "two_knockout_victory", "reason_player": "", "red_reward": 1.0, "blue_reward": -1.0, "decisions": 80, "red_knockouts": 2, "blue_knockouts": 0, "illegal_action_attempts": 0},
            {"match_type": "cross_play", "matchup": "DQN vs PPO", "orientation": "DQN_red", "algorithm_red": "DQN", "algorithm_blue": "PPO", "winner": "Blue", "terminal": True, "step_cap_reached": False, "terminal_reason": "deck_out", "reason_player": "Red", "red_reward": -1.0, "blue_reward": 1.0, "decisions": 60, "red_knockouts": 0, "blue_knockouts": 1, "illegal_action_attempts": 0},
            {"match_type": "cross_play", "matchup": "DQN vs PPO", "orientation": "DQN_red", "algorithm_red": "DQN", "algorithm_blue": "PPO", "winner": "", "terminal": False, "step_cap_reached": True, "terminal_reason": "step_cap", "reason_player": "", "red_reward": 0.0, "blue_reward": 0.0, "decisions": 200, "red_knockouts": 0, "blue_knockouts": 0, "illegal_action_attempts": 0},
        ]
        summary = _orientation_summary(records)
        assert summary["red_wins"] == summary["blue_wins"] == 1
        assert summary["nonterminal_at_step_cap"] == 1
        assert summary["red_deck_out_losses"] == 1
        matrix = _pairwise_matrix(records)
        assert matrix["DQN"]["PPO"] == 50.0
        assert matrix["PPO"]["DQN"] == 50.0



    def test_resume_does_not_duplicate_completed_games(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = TournamentConfig(games_per_orientation=2, max_decisions=2, seed=671, device="cpu", mode="self_play_only")
            output = Path(temporary_directory) / "tournament-run"
    
            def factory(algorithm: str, player_idx: int, device: torch.device):
                assert algorithm in ALGORITHM_NAMES
                return self.EndTurnAgent()
    
            first_dir, first_summary = run_tournament(
                config, output_directory=output, resume=False, agent_factory=factory
            )
            assert first_dir == output.resolve()
            assert first_summary["games_completed"] == 16
            _, resumed_summary = run_tournament(
                config, output_directory=output, resume=True, agent_factory=factory
            )
            assert resumed_summary["games_completed"] == 16
            with (output / "game_level_results.csv").open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            assert len(rows) == 16
            assert len({(row["match_type"], row["matchup"], row["orientation"], row["seed"]) for row in rows}) == 16
            for filename in ("tournament_summary.json", "tournament_summary.csv", "pairwise_win_matrix.csv", "self_play_results.csv", "tournament_report.txt", "tournament_config.json"):
                assert (output / filename).is_file()
