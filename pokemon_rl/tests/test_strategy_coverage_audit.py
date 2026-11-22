"""Focused non-training checks for optional Deep CFR strategy coverage metadata."""

from __future__ import annotations

import math
import tempfile
import json
from pathlib import Path

import numpy as np

from pokemon_rl.agents.deep_cfr import ReservoirBuffer, StrategyCoverageAudit
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM
from pokemon_rl.training.train_deep_cfr import DeepCFRConfig


def _sample(observation_value: float, target_68: float = 0.5):
    observation = np.full(OBSERVATION_DIM, observation_value, dtype=np.float32)
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    target[68] = target_68
    target[69] = 1.0 - target_68
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    mask[[68, 69]] = 1.0
    return observation, target, mask


def test_strategy_coverage_audit() -> None:
    assert DeepCFRConfig().strategy_coverage_audit is False

    audit = StrategyCoverageAudit(reservoir_capacity=2)
    observation, target_a, mask = _sample(1.0, 0.75)
    original_observation = observation.copy()
    original_target = target_a.copy()
    original_mask = mask.copy()
    assert audit.observation_hash(observation) == audit.observation_hash(observation.copy())
    assert audit.action_mask_hash(mask) == audit.action_mask_hash(mask.copy())

    red_reservoir = ReservoirBuffer(
        capacity=2,
        seed=13,
        state_dim=OBSERVATION_DIM,
        target_dim=ACTION_DIM,
        mask_dim=ACTION_DIM,
        sample_weight=True,
    )
    blue_reservoir = ReservoirBuffer(
        capacity=2,
        seed=14,
        state_dim=OBSERVATION_DIM,
        target_dim=ACTION_DIM,
        mask_dim=ACTION_DIM,
        sample_weight=True,
    )

    # The audit observes samples but leaves supplied arrays untouched.
    red_slot_zero = red_reservoir.add((observation, target_a, mask, 1.0))
    audit.record_strategy_sample(
        player=0,
        depth=0,
        observation=observation,
        target=target_a,
        action_mask=mask,
        retained_slot=red_slot_zero,
    )
    assert np.array_equal(observation, original_observation)
    assert np.array_equal(target_a, original_target)
    assert np.array_equal(mask, original_mask)

    _, target_b, _ = _sample(1.0, 0.25)
    red_slot_one = red_reservoir.add((observation, target_b, mask, 2.0))
    audit.record_strategy_sample(
        player=0,
        depth=1,
        observation=observation,
        target=target_b,
        action_mask=mask,
        retained_slot=red_slot_one,
    )
    blue_observation, blue_target, blue_mask = _sample(2.0, 0.5)
    blue_slot_zero = blue_reservoir.add((blue_observation, blue_target, blue_mask, 3.0))
    audit.record_strategy_sample(
        player=1,
        depth=2,
        observation=blue_observation,
        target=blue_target,
        action_mask=blue_mask,
        retained_slot=blue_slot_zero,
    )

    report = audit.summary()
    red = report["players"]["red"]
    blue = report["players"]["blue"]
    assert red["stream"]["total_strategy_samples_inserted"] == 2
    assert blue["stream"]["total_strategy_samples_inserted"] == 1
    assert red["final_reservoir"]["samples_by_depth"] == {"0": 1, "1": 1}
    assert blue["final_reservoir"]["samples_by_depth"] == {"2": 1}
    assert red["target_disagreement"]["repeated_observation_mask_keys"] == 1
    assert red["target_disagreement"]["keys_with_nonzero_target_variation"] == 1
    assert math.isfinite(red["target_disagreement"]["mean_target_variance_across_repeated_keys"])
    assert red["stream"]["mask_distribution"]["forced_promotion_mask_occurrences"] == 2
    assert red["stream"]["mask_distribution"]["two_action_mask_occurrences"] == 2

    # Algorithm R remains responsible for retention; the audit receives only
    # its returned slot and report writing does not change any stored tuple.
    assert red_slot_zero == 0
    assert red_slot_one == 1
    assert blue_slot_zero == 0
    assert len(red_reservoir.buffer[0]) == 4
    assert np.array_equal(red_reservoir.buffer[0][0], observation)
    assert np.array_equal(red_reservoir.buffer[0][1], target_a)
    reservoir_before_report = {
        "red": [tuple(value.copy() if isinstance(value, np.ndarray) else value for value in sample)
                for sample in red_reservoir.buffer],
        "blue": [tuple(value.copy() if isinstance(value, np.ndarray) else value for value in sample)
                 for sample in blue_reservoir.buffer],
    }

    with tempfile.TemporaryDirectory(prefix="deep_cfr_coverage_audit_") as directory:
        paths = audit.write_report(
            directory,
            strategy_reservoirs={0: red_reservoir, 1: blue_reservoir},
        )
        assert all(path.endswith((".json", ".txt", ".jsonl")) for path in paths.values())
        assert paths["final_reservoir_index_jsonl"].endswith("strategy_coverage_final_reservoir_index.jsonl")
        with open(paths["final_reservoir_index_jsonl"], encoding="utf-8") as index_file:
            index_records = [json.loads(line) for line in index_file]

        assert [(record["player"], record["reservoir_slot"]) for record in index_records] == [
            ("red", 0), ("red", 1), ("blue", 0),
        ]
        assert index_records[0]["traversal_depth"] == 0
        assert index_records[1]["traversal_depth"] == 1
        assert index_records[2]["traversal_depth"] == 2
        for record, sample in zip(
            index_records,
            [red_reservoir.buffer[0], red_reservoir.buffer[1], blue_reservoir.buffer[0]],
        ):
            assert record["observation_hash"] == audit.observation_hash(sample[0]).hex()
            assert record["action_mask_hash"] == audit.action_mask_hash(sample[2]).hex()
            assert record["legal_action_count"] == int(np.count_nonzero(sample[2]))

        repeat_paths = audit.write_report(
            str(Path(directory) / "repeat"),
            strategy_reservoirs={0: red_reservoir, 1: blue_reservoir},
        )
        assert Path(paths["final_reservoir_index_jsonl"]).read_text(encoding="utf-8") == Path(
            repeat_paths["final_reservoir_index_jsonl"]
        ).read_text(encoding="utf-8")

    for player_name, reservoir in (("red", red_reservoir), ("blue", blue_reservoir)):
        for expected, actual in zip(reservoir_before_report[player_name], reservoir.buffer):
            for expected_value, actual_value in zip(expected, actual):
                if isinstance(expected_value, np.ndarray):
                    assert np.array_equal(expected_value, actual_value)
                else:
                    assert expected_value == actual_value


if __name__ == "__main__":
    test_strategy_coverage_audit()
