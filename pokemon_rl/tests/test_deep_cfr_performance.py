"""One-traversal performance validation on the real 20-card production deck.

This is intentionally not an experiment runner: it creates exactly one Red and
one Blue external-sampling traversal, prints their cost profile, and verifies
their samples remain in their respective reservoirs.
"""

from __future__ import annotations

import math
import tempfile

import torch

from pokemon_rl.training.train_deep_cfr import BLUE, RED, DeepCFRConfig, DeepCFRTrainer


PRODUCTION_TRAVERSAL_HORIZON = 4


def _print_profile(label: str, profile: dict) -> None:
    print(f"\n{label} traversal performance profile")
    print("-" * 70)
    print(f"  Wall-clock time:             {profile['elapsed_seconds']:.3f} s")
    print(f"  Recursive calls:             {profile['recursive_calls']}")
    print(f"  Maximum expanded depth:      {profile['max_depth']}")
    print(f"  GameState deep copies:       {profile['state_copies']}")
    print(f"  GameEngine transitions:      {profile['game_transitions']}")
    print(f"  Decision nodes:              {profile['decision_nodes']}")
    print(f"    Traverser / opponent:      {profile['traverser_decision_nodes']} / {profile['opponent_decision_nodes']}")
    print(f"  Terminal leaves:             {profile['terminal_nodes']}")
    print(f"  Average legal-action count:  {profile['average_legal_action_count']:.2f}")
    print(f"  Horizon continuations:       {profile['horizon_rollouts']}")
    print(f"  Continuation rollout steps:  {profile['rollout_steps']}")
    print(f"  Root utility:                {profile['root_utility']:+.3f}")


def test_deep_cfr_real_deck_traversal_performance() -> None:
    print("=" * 70)
    print("   STARTING REAL 20-CARD DEEP CFR ONE-TRAVERSAL PERFORMANCE TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="deep_cfr_performance_") as checkpoint_dir:
        config = DeepCFRConfig(
            num_iterations=1,
            traversals_per_iter=1,
            batch_size=16,
            reservoir_capacity=1_000,
            regret_epochs=1,
            strategy_epochs=1,
            max_steps=PRODUCTION_TRAVERSAL_HORIZON,
            checkpoint_dir=checkpoint_dir,
            seed=9_001,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compact_test_deck=False,
        )
        trainer = DeepCFRTrainer(config)
        print(f"Device: {trainer.device}; real 20-card deck; hard expanded-tree horizon: {config.max_steps}")

        red = trainer._run_player_traversals(iteration=1, traverser=RED, verbose=True)
        blue = trainer._run_player_traversals(iteration=1, traverser=BLUE, verbose=True)
        red_profile = trainer.traversal_records[0]["diagnostics"]
        blue_profile = trainer.traversal_records[1]["diagnostics"]
        _print_profile("Red", red_profile)
        _print_profile("Blue", blue_profile)

        assert math.isfinite(red["utility"]) and -1.0 <= red["utility"] <= 1.0
        assert math.isfinite(blue["utility"]) and -1.0 <= blue["utility"] <= 1.0
        assert red_profile["decision_nodes"] > 0 and blue_profile["decision_nodes"] > 0
        assert red_profile["recursive_calls"] >= red_profile["decision_nodes"]
        assert blue_profile["recursive_calls"] >= blue_profile["decision_nodes"]
        assert red_profile["max_depth"] <= config.max_steps
        assert blue_profile["max_depth"] <= config.max_steps
        assert red_profile["elapsed_seconds"] > 0.0 and blue_profile["elapsed_seconds"] > 0.0
        assert trainer.regret_buffers[RED].num_samples_seen == red["regret_samples"]
        assert trainer.strategy_buffers[RED].num_samples_seen == red["strategy_samples"]
        assert trainer.regret_buffers[BLUE].num_samples_seen == blue["regret_samples"]
        assert trainer.strategy_buffers[BLUE].num_samples_seen == blue["strategy_samples"]
        assert trainer.regret_buffers[RED].buffer is not trainer.regret_buffers[BLUE].buffer
        assert trainer.strategy_buffers[RED].buffer is not trainer.strategy_buffers[BLUE].buffer
        print("\nFinite utilities, 20-card traversal profiles, and Red/Blue memory separation: PASSED")

    print("=" * 70)
    print("        REAL-DECK ONE-RED/ONE-BLUE PERFORMANCE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_deep_cfr_real_deck_traversal_performance()
