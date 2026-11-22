"""Configuration and CLI checks for isolated Deep CFR experiment pilots."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

import run_deep_cfr_experiment as experiment
from pokemon_rl.training.train_deep_cfr import DeepCFRConfig


def test_iteration_weighted_depth6_pilot_cli_flag() -> None:
    with patch.object(
        sys,
        "argv",
        ["run_deep_cfr_experiment.py", "--iteration-weighted-depth6-pilot", "--strategy-coverage-audit"],
    ):
        arguments = experiment._parse_args()

    assert arguments.iteration_weighted_depth6_pilot is True
    assert arguments.strategy_coverage_audit is True


def test_iteration_weighted_depth6_pilot_configuration_and_isolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiment, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(experiment, "print_hardware_info", lambda: torch.device("cpu"))
    captured: dict[str, object] = {}

    def fake_train(config: DeepCFRConfig, **_kwargs):
        captured["config"] = config
        return SimpleNamespace(traversal_records=[]), {"iteration": [1]}

    def capture_pilot_summary(_history, _trainer, _training_time, results_dir, **kwargs):
        captured["summary_dir"] = results_dir
        captured["summary_options"] = kwargs

    monkeypatch.setattr(experiment, "train_deep_cfr", fake_train)
    monkeypatch.setattr(experiment, "_print_horizon_pilot_summary", capture_pilot_summary)
    monkeypatch.setattr(
        experiment,
        "evaluate_against_random_blue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pilot must skip evaluation")),
    )

    experiment.main(
        iteration_weighted_depth6_pilot=True,
        strategy_coverage_audit=True,
    )

    config = captured["config"]
    assert isinstance(config, DeepCFRConfig)
    assert config.num_iterations == 10
    assert config.traversals_per_iter == 16
    assert config.max_steps == 6
    assert config.batch_size == 128
    assert config.reservoir_capacity == 40_000
    assert config.learning_rate == experiment.LEARNING_RATE
    assert config.regret_epochs == experiment.REGRET_EPOCHS
    assert config.strategy_epochs == experiment.STRATEGY_EPOCHS
    assert config.seed == experiment.BASE_SEED
    assert config.compact_test_deck is False
    assert config.strategy_coverage_audit is True

    expected_output_dir = os.path.join(
        str(tmp_path), "results", "deep_cfr_iteration_weighted_depth6_pilot"
    )
    assert config.checkpoint_dir == expected_output_dir
    assert captured["summary_dir"] == expected_output_dir
    assert captured["summary_options"] == {
        "title": "DEEP CFR ITERATION-WEIGHTED DEPTH-6 PILOT SUMMARY",
        "explicit_horizon": 6,
    }
