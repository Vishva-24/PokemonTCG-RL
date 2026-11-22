"""Rapid end-to-end validation for the true two-player Deep CFR loop."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, Iterable, List

import torch

from pokemon_rl.agents.deep_cfr import RegretNetwork, StrategyNetwork
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM
from pokemon_rl.training.train_deep_cfr import BLUE, RED, DeepCFRConfig, DeepCFRTrainer


def _parameters_changed(before: Iterable[torch.Tensor], network: torch.nn.Module) -> bool:
    return any(not torch.equal(old, new.detach().cpu()) for old, new in zip(before, network.parameters()))


def _diagnostic_snapshot(history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Exclude wall-clock measurements, which cannot be deterministic."""
    return {
        key: history[key]
        for key in (
            "red_traversal_utility", "blue_traversal_utility",
            "red_decision_nodes", "blue_decision_nodes",
            "red_terminal_nodes", "blue_terminal_nodes",
            "red_regret_samples", "blue_regret_samples",
            "red_strategy_samples", "blue_strategy_samples",
            "red_regret_network_loss", "blue_regret_network_loss",
            "red_strategy_network_loss", "blue_strategy_network_loss",
            "red_regret_reservoir_size", "blue_regret_reservoir_size",
            "red_strategy_reservoir_size", "blue_strategy_reservoir_size",
        )
    }


def _assert_checkpoint_round_trip(trainer: DeepCFRTrainer, checkpoint_dir: str) -> None:
    """Verify all required files exist, load, and final weights round-trip."""
    for stage in ("best", "final"):
        for player, player_name in ((RED, "red"), (BLUE, "blue")):
            for network_name, network_type, trained_network in (
                ("strategy", StrategyNetwork, trainer.strategy_networks[player]),
                ("regret", RegretNetwork, trainer.regret_networks[player]),
            ):
                path = trainer._checkpoint_path(stage, player, network_name)
                assert os.path.exists(path), f"Missing checkpoint: {path}"
                loaded = network_type(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to("cpu")
                loaded.device = torch.device("cpu")
                loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                if stage == "final":
                    for loaded_param, trained_param in zip(loaded.parameters(), trained_network.parameters()):
                        assert torch.equal(loaded_param, trained_param.detach().cpu()), "Final checkpoint did not round-trip."


def test_train_deep_cfr_pipeline() -> None:
    print("=" * 70)
    print("           STARTING RAPID TRUE TWO-PLAYER DEEP CFR TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="deep_cfr_test_") as first_dir, tempfile.TemporaryDirectory(prefix="deep_cfr_repro_") as second_dir:
        # The requested rapid configuration is explicitly CPU-forced for the CPU
        # execution check, even on a CUDA-equipped machine.
        config = DeepCFRConfig.rapid(checkpoint_dir=first_dir, seed=7_321)
        config.device = "cpu"
        assert (config.num_iterations, config.traversals_per_iter, config.batch_size) == (2, 4, 16)
        assert config.reservoir_capacity == 1_000
        assert (config.regret_epochs, config.strategy_epochs) == (2, 2)
        assert config.compact_test_deck is True

        trainer = DeepCFRTrainer(config)
        assert trainer.regret_networks[RED] is not trainer.regret_networks[BLUE]
        assert trainer.strategy_networks[RED] is not trainer.strategy_networks[BLUE]
        assert trainer.regret_buffers[RED] is not trainer.regret_buffers[BLUE]
        assert trainer.strategy_buffers[RED] is not trainer.strategy_buffers[BLUE]
        assert all(next(network.parameters()).device.type == "cpu" for network in trainer.regret_networks.values())
        assert all(next(network.parameters()).device.type == "cpu" for network in trainer.strategy_networks.values())
        assert all(
            network.net[0].in_features == OBSERVATION_DIM and network.net[-1].out_features == ACTION_DIM
            for networks in (trainer.regret_networks, trainer.strategy_networks)
            for network in networks.values()
        )
        print("Initialization, separate Red/Blue networks, memories, and CPU placement: PASSED")

        before = {
            (network_kind, player): [parameter.detach().cpu().clone() for parameter in network.parameters()]
            for network_kind, networks in (("regret", trainer.regret_networks), ("strategy", trainer.strategy_networks))
            for player, network in networks.items()
        }
        # Progress appears after each CFR role, so a genuine long traversal is
        # observable rather than looking frozen in an interactive terminal.
        history = trainer.train(verbose=True, show_hardware=False)

        # Both roles must traverse four times in each of two outer iterations.
        assert len(history["iteration"]) == 2
        assert len(trainer.traversal_records) == 16
        assert sum(record["traverser"] == RED for record in trainer.traversal_records) == 8
        assert sum(record["traverser"] == BLUE for record in trainer.traversal_records) == 8
        assert all(record["diagnostics"]["decision_nodes"] > 0 for record in trainer.traversal_records)
        assert all(record["diagnostics"]["terminal_nodes"] > 0 for record in trainer.traversal_records)
        print("Four Red and four Blue self-play traversals per iteration: PASSED")

        # Each scalar buffer is passed only to the matching traverser. Seen counts
        # must exactly equal the diagnostics for that player, proving no crossover.
        for player, label in ((RED, "red"), (BLUE, "blue")):
            expected_regret = sum(history[f"{label}_regret_samples"])
            expected_strategy = sum(history[f"{label}_strategy_samples"])
            assert trainer.regret_buffers[player].num_samples_seen == expected_regret
            assert trainer.strategy_buffers[player].num_samples_seen == expected_strategy
            assert 0 < len(trainer.regret_buffers[player]) <= config.reservoir_capacity
            assert 0 < len(trainer.strategy_buffers[player]) <= config.reservoir_capacity
            strategy_sample = trainer.strategy_buffers[player].sample(1)[0]
            assert len(strategy_sample) == 4
            _, strategy_target, strategy_mask, iteration_weight = strategy_sample
            assert strategy_target.shape == (ACTION_DIM,)
            assert strategy_mask.shape == (ACTION_DIM,)
            assert math.isfinite(float(iteration_weight))
            assert 1.0 <= float(iteration_weight) <= float(config.num_iterations)
            assert torch.all((torch.from_numpy(strategy_mask) == 0.0) | (torch.from_numpy(strategy_mask) == 1.0))
            assert float(strategy_mask.sum()) > 0.0
            assert torch.all(torch.from_numpy(strategy_target[strategy_mask == 0.0]) == 0.0)
            assert abs(float(strategy_target[strategy_mask == 1.0].sum()) - 1.0) < 1e-5
        assert trainer.regret_buffers[RED].buffer is not trainer.regret_buffers[BLUE].buffer
        assert trainer.strategy_buffers[RED].buffer is not trainer.strategy_buffers[BLUE].buffer
        print("Player-specific reservoir insertion and capacity limits: PASSED")

        for key, values in history.items():
            if "loss" in key or "utility" in key:
                assert all(math.isfinite(float(value)) for value in values), f"Non-finite value in {key}"
        for key in ("red_traversal_utility", "blue_traversal_utility"):
            assert all(-1.0 <= value <= 1.0 for value in history[key]), f"Out-of-range utility in {key}"
        assert all(history[key][-1] > 0.0 for key in (
            "red_regret_network_loss", "blue_regret_network_loss",
            "red_strategy_network_loss", "blue_strategy_network_loss",
        ))
        assert all(
            _parameters_changed(before[(kind, player)], networks[player])
            for kind, networks in (("regret", trainer.regret_networks), ("strategy", trainer.strategy_networks))
            for player in (RED, BLUE)
        )
        print("Finite losses/utilities and regret/strategy parameter updates: PASSED")

        metrics_path = os.path.join(first_dir, "deep_cfr_training_metrics.csv")
        assert os.path.exists(metrics_path)
        _assert_checkpoint_round_trip(trainer, first_dir)
        print("Required metrics and best/final checkpoints save and load: PASSED")

        # Re-run the exact rapid configuration with the same seed and compare
        # deterministic diagnostics, excluding only elapsed wall-clock time.
        reproducible_config = DeepCFRConfig.rapid(checkpoint_dir=second_dir, seed=7_321)
        reproducible_config.device = "cpu"
        reproducible_history = DeepCFRTrainer(reproducible_config).train(verbose=False)
        assert _diagnostic_snapshot(history) == _diagnostic_snapshot(reproducible_history)
        print("Repeated seed produces identical deterministic diagnostics: PASSED")

        if torch.cuda.is_available():
            cuda_config = DeepCFRConfig.rapid(checkpoint_dir=second_dir, seed=7_321)
            cuda_config.device = "cuda"
            cuda_trainer = DeepCFRTrainer(cuda_config)
            assert all(next(network.parameters()).device.type == "cuda" for network in cuda_trainer.regret_networks.values())
            assert all(next(network.parameters()).device.type == "cuda" for network in cuda_trainer.strategy_networks.values())
            print("CUDA networks initialize on GPU when available: PASSED")
        else:
            print("CUDA unavailable; CUDA placement check skipped as expected.")

    print("=" * 70)
    print("                RAPID DEEP CFR TRAINING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_train_deep_cfr_pipeline()
