"""Focused non-training checks for StrategyNetwork evaluation coverage metadata."""

from __future__ import annotations

import json
import math
import tempfile

import numpy as np

from pokemon_rl.agents.deep_cfr import StrategyCoverageAudit
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM
from run_deep_cfr_experiment import StrategyEvaluationCoverageAudit


def _state(value: float, legal_actions: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    observation = np.full(OBSERVATION_DIM, value, dtype=np.float32)
    action_mask = np.zeros(ACTION_DIM, dtype=np.float32)
    action_mask[list(legal_actions)] = 1.0
    return observation, action_mask


def test_strategy_evaluation_coverage_audit() -> None:
    observation, promotion_mask = _state(1.0, (68, 69))
    mask_variant = np.zeros(ACTION_DIM, dtype=np.float32)
    mask_variant[68] = 1.0
    observation_hash = StrategyCoverageAudit.observation_hash(observation).hex()
    promotion_hash = StrategyCoverageAudit.action_mask_hash(promotion_mask).hex()

    audit = StrategyEvaluationCoverageAudit(
        training_observation_hashes={observation_hash},
        training_observation_mask_hashes={(observation_hash, promotion_hash)},
        training_coverage_index_path="training/index.jsonl",
    )
    original_observation = observation.copy()
    original_mask = promotion_mask.copy()
    audit.record_decision(
        policy_mode="greedy",
        observation=observation,
        action_mask=promotion_mask,
        game_step=0,
        turn_count=1,
        selected_action=68,
        selected_action_category="promote",
        forced_promotion=True,
    )
    # A duplicate decision is counted as a second occurrence, not a second
    # unique state, and auditing never mutates policy inputs.
    audit.record_decision(
        policy_mode="greedy",
        observation=observation,
        action_mask=promotion_mask,
        game_step=1,
        turn_count=1,
        selected_action=69,
        selected_action_category="promote",
        forced_promotion=True,
    )
    audit.record_decision(
        policy_mode="sampled",
        observation=observation,
        action_mask=mask_variant,
        game_step=5,
        turn_count=2,
        selected_action=68,
        selected_action_category="promote",
        forced_promotion=True,
    )
    assert np.array_equal(observation, original_observation)
    assert np.array_equal(promotion_mask, original_mask)

    summary = audit.summary()
    greedy = summary["policies"]["greedy"]
    sampled = summary["policies"]["sampled"]
    assert greedy["evaluation_decision_occurrences"]["decision_occurrences"] == 2
    assert greedy["unique_evaluation_observation_mask_hashes"] == 1
    assert greedy["evaluation_decision_occurrences"]["reservoir_covered_occurrences"] == 2
    assert sampled["evaluation_decision_occurrences"]["unseen_occurrences"] == 1
    assert sampled["evaluation_decision_occurrences"]["observation_seen_mask_unseen_occurrences"] == 1
    assert sampled["coverage_by_game_step_range"]["5-9"]["decision_occurrences"] == 1
    assert math.isfinite(greedy["unique_observation_mask_coverage"]["reservoir_covered_percent"])

    with tempfile.TemporaryDirectory(prefix="strategy_eval_coverage_") as directory:
        paths = audit.write_report(directory)
        assert paths["summary_json"].endswith(".json")
        assert paths["summary_text"].endswith(".txt")
        with open(paths["summary_json"], "r", encoding="utf-8") as json_file:
            loaded = json.load(json_file)
        with open(paths["summary_json"], "rb") as json_file:
            first_serialization = json_file.read()

        # Tuple-keyed occurrence data round-trips as explicit, non-ambiguous
        # hash records. Duplicate greedy occurrences are retained exactly.
        greedy_records = loaded["policies"]["greedy"]["evaluation_observation_mask_occurrences"]
        assert greedy_records == [{
            "action_mask_hash": promotion_hash,
            "count": 2,
            "observation_hash": observation_hash,
        }]
        assert loaded["policies"]["sampled"]["evaluation_decision_occurrences"]["unseen_occurrences"] == 1
        assert loaded["policies"]["greedy"]["evaluation_observation_occurrences"][observation_hash] == 2

        # Rewriting unchanged aggregate diagnostics must produce byte-for-byte
        # identical JSON, including deterministic record ordering.
        audit.write_report(directory)
        with open(paths["summary_json"], "rb") as json_file:
            assert json_file.read() == first_serialization


if __name__ == "__main__":
    test_strategy_evaluation_coverage_audit()
