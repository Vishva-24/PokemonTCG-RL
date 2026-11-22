"""Run the 100-iteration Deep CFR experiment and its 1,500-game benchmark.

This script is intentionally not imported by the rapid unit test.  Training is
two-player CFR self-play; only the final evaluation uses ``PokemonTCGEnv`` and
its established built-in Random Blue opponent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pokemon_rl.agents.deep_cfr import (
    RegretNetwork,
    StrategyCoverageAudit,
    StrategyNetwork,
    regret_matching,
)
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv
from pokemon_rl.training.train_deep_cfr import DeepCFRConfig, train_deep_cfr


# Keep the real-experiment controls in the runner, like the DQN/PPO scripts.
NUM_ITERATIONS = 100
TRAVERSALS_PER_ITER = 128
BATCH_SIZE = 128
RESERVOIR_CAPACITY = 40_000
LEARNING_RATE = 1e-4
REGRET_EPOCHS = 32
STRATEGY_EPOCHS = 32
# Measured depth-limited MCCFR horizon. At the boundary, the traversal rolls
# out the current regret-matching profile to a genuine terminal payoff rather
# than assigning a fabricated value. See test_deep_cfr_performance.py.
MAX_STEPS = 4
BASE_SEED = 4_000

# Controlled horizon pilot.  Keep it separate from the full 100-iteration
# baseline so its fresh checkpoints and metrics cannot overwrite the baseline.
PILOT_NUM_ITERATIONS = 10
PILOT_TRAVERSALS_PER_ITER = 16
PILOT_MAX_STEPS = 8
PILOT_RESULTS_DIRECTORY = "deep_cfr_horizon8_pilot"

# Controlled strategy-averaging pilot. This isolates iteration-weighted
# StrategyNetwork fitting at the validated production horizon without touching
# the 100-iteration baseline or the earlier horizon-8 pilot.
ITERATION_WEIGHTED_PILOT_NUM_ITERATIONS = 10
ITERATION_WEIGHTED_PILOT_TRAVERSALS_PER_ITER = 16
ITERATION_WEIGHTED_PILOT_MAX_STEPS = 4
ITERATION_WEIGHTED_PILOT_RESULTS_DIRECTORY = "deep_cfr_iteration_weighted_depth4_pilot"

# Isolated iteration-weighted horizon-6 pilot. It uses the depth-4 pilot's
# training settings while writing all checkpoints and metrics to its own folder.
ITERATION_WEIGHTED_DEPTH6_PILOT_NUM_ITERATIONS = 10
ITERATION_WEIGHTED_DEPTH6_PILOT_TRAVERSALS_PER_ITER = 16
ITERATION_WEIGHTED_DEPTH6_PILOT_MAX_STEPS = 6
ITERATION_WEIGHTED_DEPTH6_PILOT_RESULTS_DIRECTORY = "deep_cfr_iteration_weighted_depth6_pilot"

# One-iteration throughput measurement.  This is deliberately separate from
# both the full horizon-4 experiment and the 10-iteration horizon-8 pilot.
RUNTIME_TEST_NUM_ITERATIONS = 1
RUNTIME_TEST_TRAVERSALS_PER_ITER = 128
RUNTIME_TEST_MAX_STEPS = 6
RUNTIME_TEST_RESULTS_DIRECTORY = "deep_cfr_horizon6_runtime_test"

# Read-only coverage comparison for a supplied StrategyNetwork checkpoint.
# This mode deliberately uses the first established evaluation range only.
STRATEGY_EVALUATION_COVERAGE_SEED_RANGES: List[Tuple[int, int]] = [(10_000, 10_199)]
STRATEGY_EVALUATION_COVERAGE_MAX_STEPS = 200
STRATEGY_EVALUATION_COVERAGE_RESULTS_DIRECTORY = "strategy_evaluation_coverage_audit"

SEED_RANGES: List[Tuple[int, int]] = [
    (10_000, 10_499),
    (20_000, 20_499),
    (30_000, 30_499),
]


# These groups match the approved Phase 2 70-action mapping.  They are used
# only to report how an evaluation policy behaves; they do not affect training
# or environment action decoding.
_ACTION_GROUPS = (
    "end_turn",
    "attack",
    "retreat",
    "play_pokemon",
    "attach_energy",
    "play_potion",
    "promote",
)
_EVALUATION_POLICIES = {"random", "greedy", "sampled", "regret_matching"}


def _coverage_counts_template() -> Dict[str, int]:
    """Create occurrence counters for one passive coverage grouping."""
    return {
        "decision_occurrences": 0,
        "reservoir_covered_occurrences": 0,
        "unseen_occurrences": 0,
        "observation_seen_mask_unseen_occurrences": 0,
    }


class StrategyEvaluationCoverageAudit:
    """Passive membership trace for evaluation states against a final reservoir.

    Hashes are deterministic fingerprints of the current 50-D observation and
    70-D legal-action mask. They are not claimed to be canonical information
    set IDs. The evaluator calls :meth:`record_decision` only after selecting a
    legal action, so this class cannot change policy behavior.
    """

    def __init__(
        self,
        *,
        training_observation_hashes: set[str],
        training_observation_mask_hashes: set[Tuple[str, str]],
        training_coverage_index_path: str,
    ) -> None:
        self.training_observation_hashes = set(training_observation_hashes)
        self.training_observation_mask_hashes = set(training_observation_mask_hashes)
        self.training_coverage_index_path = training_coverage_index_path
        self._policies: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _new_policy_data() -> Dict[str, Any]:
        return {
            "decision_occurrences": 0,
            "reservoir_covered_occurrences": 0,
            "unseen_occurrences": 0,
            "observation_seen_mask_unseen_occurrences": 0,
            "observation_occurrences": {},
            "observation_mask_occurrences": {},
            "step_ranges": {},
            "legal_action_counts": {},
            "selected_action_indices": {},
            "selected_action_categories": {},
            "legal_action_families": {},
            "forced_promotion_states": _coverage_counts_template(),
            "turn_count_histogram": {},
        }

    @staticmethod
    def _step_range(game_step: int) -> str:
        if game_step <= 4:
            return "0-4"
        if game_step <= 9:
            return "5-9"
        if game_step <= 19:
            return "10-19"
        if game_step <= 39:
            return "20-39"
        return "40+"

    @staticmethod
    def _update_counter_group(
        groups: Dict[str, Dict[str, int]],
        group: str,
        *,
        reservoir_covered: bool,
        observation_seen_mask_unseen: bool,
    ) -> None:
        counts = groups.setdefault(group, _coverage_counts_template())
        counts["decision_occurrences"] += 1
        if reservoir_covered:
            counts["reservoir_covered_occurrences"] += 1
        else:
            counts["unseen_occurrences"] += 1
        if observation_seen_mask_unseen:
            counts["observation_seen_mask_unseen_occurrences"] += 1

    @staticmethod
    def _record_occurrence(
        data: Dict[str, Any], *, reservoir_covered: bool, observation_seen_mask_unseen: bool
    ) -> None:
        data["decision_occurrences"] += 1
        if reservoir_covered:
            data["reservoir_covered_occurrences"] += 1
        else:
            data["unseen_occurrences"] += 1
        if observation_seen_mask_unseen:
            data["observation_seen_mask_unseen_occurrences"] += 1

    def record_decision(
        self,
        *,
        policy_mode: str,
        observation: np.ndarray,
        action_mask: np.ndarray,
        game_step: int,
        turn_count: int,
        selected_action: int,
        selected_action_category: str,
        forced_promotion: bool,
    ) -> None:
        """Observe one selected evaluation action without changing it."""
        if policy_mode not in {"greedy", "sampled"}:
            raise ValueError("Strategy evaluation coverage supports greedy and sampled policies only.")
        if np.asarray(observation).shape != (OBSERVATION_DIM,):
            raise ValueError("Evaluation coverage observations must be 50-D.")
        if np.asarray(action_mask).shape != (ACTION_DIM,):
            raise ValueError("Evaluation coverage masks must have 70 actions.")
        if not 0 <= selected_action < ACTION_DIM or action_mask[selected_action] != 1.0:
            raise ValueError("Evaluation coverage must observe an already-selected legal action.")
        if game_step < 0 or turn_count < 0:
            raise ValueError("Evaluation coverage step and turn counts must be non-negative.")

        data = self._policies.setdefault(policy_mode, self._new_policy_data())
        observation_hash = StrategyCoverageAudit.observation_hash(observation).hex()
        mask_hash = StrategyCoverageAudit.action_mask_hash(action_mask).hex()
        observation_mask_key = (observation_hash, mask_hash)
        reservoir_covered = observation_mask_key in self.training_observation_mask_hashes
        observation_seen = observation_hash in self.training_observation_hashes
        observation_seen_mask_unseen = observation_seen and not reservoir_covered
        legal_action_count = int(np.count_nonzero(action_mask))

        self._record_occurrence(
            data,
            reservoir_covered=reservoir_covered,
            observation_seen_mask_unseen=observation_seen_mask_unseen,
        )
        data["observation_occurrences"][observation_hash] = (
            data["observation_occurrences"].get(observation_hash, 0) + 1
        )
        data["observation_mask_occurrences"][observation_mask_key] = (
            data["observation_mask_occurrences"].get(observation_mask_key, 0) + 1
        )
        self._update_counter_group(
            data["step_ranges"],
            self._step_range(game_step),
            reservoir_covered=reservoir_covered,
            observation_seen_mask_unseen=observation_seen_mask_unseen,
        )
        self._update_counter_group(
            data["legal_action_counts"],
            str(legal_action_count),
            reservoir_covered=reservoir_covered,
            observation_seen_mask_unseen=observation_seen_mask_unseen,
        )
        self._update_counter_group(
            data["selected_action_categories"],
            selected_action_category,
            reservoir_covered=reservoir_covered,
            observation_seen_mask_unseen=observation_seen_mask_unseen,
        )
        data["selected_action_indices"][str(selected_action)] = (
            data["selected_action_indices"].get(str(selected_action), 0) + 1
        )
        for action_family in {_action_group(int(index)) for index in np.flatnonzero(action_mask)}:
            self._update_counter_group(
                data["legal_action_families"],
                action_family,
                reservoir_covered=reservoir_covered,
                observation_seen_mask_unseen=observation_seen_mask_unseen,
            )
        if forced_promotion:
            forced_counts = data["forced_promotion_states"]
            forced_counts["decision_occurrences"] += 1
            if reservoir_covered:
                forced_counts["reservoir_covered_occurrences"] += 1
            else:
                forced_counts["unseen_occurrences"] += 1
            if observation_seen_mask_unseen:
                forced_counts["observation_seen_mask_unseen_occurrences"] += 1
        data["turn_count_histogram"][str(turn_count)] = (
            data["turn_count_histogram"].get(str(turn_count), 0) + 1
        )

    @staticmethod
    def _with_percentages(counts: Dict[str, int]) -> Dict[str, Any]:
        """Return one JSON-safe coverage counter group and its percentages.

        ``_new_policy_data`` also owns internal maps keyed by observation and
        observation-plus-mask fingerprints.  Those maps are deliberately not
        expanded here: the latter uses tuple keys and is reported separately as
        structured records by :meth:`_policy_summary`.
        """
        scalar_counts = {
            "decision_occurrences": int(counts["decision_occurrences"]),
            "reservoir_covered_occurrences": int(counts["reservoir_covered_occurrences"]),
            "unseen_occurrences": int(counts["unseen_occurrences"]),
            "observation_seen_mask_unseen_occurrences": int(
                counts["observation_seen_mask_unseen_occurrences"]
            ),
        }
        total = scalar_counts["decision_occurrences"]
        denominator = float(total) if total else 1.0
        return {
            **scalar_counts,
            "reservoir_covered_percent": (
                100.0 * scalar_counts["reservoir_covered_occurrences"] / denominator
            ),
            "unseen_percent": 100.0 * scalar_counts["unseen_occurrences"] / denominator,
            "observation_seen_mask_unseen_percent": (
                100.0 * scalar_counts["observation_seen_mask_unseen_occurrences"] / denominator
            ),
        }

    @staticmethod
    def _observation_mask_occurrence_records(
        occurrences: Dict[Tuple[str, str], int]
    ) -> List[Dict[str, Any]]:
        """Encode tuple-keyed coverage counts as deterministic JSON records.

        Separating the two hash components preserves the original tuple's
        structure without an ambiguous string conversion or information loss.
        """
        return [
            {
                "observation_hash": observation_hash,
                "action_mask_hash": action_mask_hash,
                "count": int(count),
            }
            for (observation_hash, action_mask_hash), count in sorted(occurrences.items())
        ]

    def _policy_summary(self, policy_mode: str, data: Dict[str, Any]) -> Dict[str, Any]:
        unique_observations = set(data["observation_occurrences"])
        unique_observation_masks = set(data["observation_mask_occurrences"])
        covered_unique_masks = unique_observation_masks & self.training_observation_mask_hashes
        observation_seen_mask_unseen_unique = {
            key for key in unique_observation_masks
            if key[0] in self.training_observation_hashes and key not in self.training_observation_mask_hashes
        }
        unique_total = len(unique_observation_masks)
        unique_denominator = float(unique_total) if unique_total else 1.0
        return {
            "policy_mode": policy_mode,
            "evaluation_decision_occurrences": self._with_percentages(data),
            "evaluation_observation_occurrences": dict(sorted(
                data["observation_occurrences"].items()
            )),
            "evaluation_observation_mask_occurrences": self._observation_mask_occurrence_records(
                data["observation_mask_occurrences"]
            ),
            "unique_evaluation_observation_hashes": len(unique_observations),
            "unique_evaluation_observation_mask_hashes": unique_total,
            "unique_observation_mask_coverage": {
                "reservoir_covered_unique_states": len(covered_unique_masks),
                "unseen_unique_states": unique_total - len(covered_unique_masks),
                "observation_seen_mask_unseen_unique_states": len(observation_seen_mask_unseen_unique),
                "reservoir_covered_percent": 100.0 * len(covered_unique_masks) / unique_denominator,
                "unseen_percent": 100.0 * (unique_total - len(covered_unique_masks)) / unique_denominator,
                "observation_seen_mask_unseen_percent": (
                    100.0 * len(observation_seen_mask_unseen_unique) / unique_denominator
                ),
            },
            "coverage_by_game_step_range": {
                key: self._with_percentages(counts) for key, counts in data["step_ranges"].items()
            },
            "coverage_by_legal_action_count": {
                key: self._with_percentages(counts) for key, counts in data["legal_action_counts"].items()
            },
            "coverage_by_selected_action_category": {
                key: self._with_percentages(counts)
                for key, counts in data["selected_action_categories"].items()
            },
            "selected_action_index_histogram": dict(sorted(
                data["selected_action_indices"].items(), key=lambda item: int(item[0])
            )),
            "coverage_by_legal_action_family": {
                key: self._with_percentages(counts)
                for key, counts in data["legal_action_families"].items()
            },
            "forced_promotion_state_coverage": self._with_percentages(data["forced_promotion_states"]),
            "turn_count_histogram": dict(sorted(data["turn_count_histogram"].items(), key=lambda item: int(item[0]))),
        }

    def summary(self) -> Dict[str, Any]:
        """Return occurrence and unique-state coverage for every traced policy."""
        return {
            "schema_version": 1,
            "terminology_note": (
                "Observation hashes and observation-plus-mask hashes are deterministic "
                "fingerprints, not guaranteed canonical information-set identifiers."
            ),
            "training_coverage_index": self.training_coverage_index_path,
            "training_final_reservoir_unique_observation_hashes": len(self.training_observation_hashes),
            "training_final_reservoir_unique_observation_mask_hashes": len(
                self.training_observation_mask_hashes
            ),
            "policies": {
                policy_mode: self._policy_summary(policy_mode, data)
                for policy_mode, data in self._policies.items()
            },
        }

    def write_report(self, output_dir: str) -> Dict[str, str]:
        """Write separate JSON and concise text coverage reports."""
        os.makedirs(output_dir, exist_ok=True)
        summary = self.summary()
        json_path = os.path.join(output_dir, "strategy_evaluation_coverage_summary.json")
        text_path = os.path.join(output_dir, "strategy_evaluation_coverage_summary.txt")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=2, sort_keys=True)
        lines = [
            "DEEP CFR STRATEGY EVALUATION COVERAGE AUDIT",
            "=" * 70,
            summary["terminology_note"],
            f"Training coverage index: {summary['training_coverage_index']}",
        ]
        for policy_mode, report in summary["policies"].items():
            occurrences = report["evaluation_decision_occurrences"]
            unique = report["unique_observation_mask_coverage"]
            lines.extend((
                "",
                policy_mode.upper(),
                f"  Evaluation decision occurrences: {occurrences['decision_occurrences']}",
                f"  Reservoir-covered occurrences: {occurrences['reservoir_covered_occurrences']} "
                f"({occurrences['reservoir_covered_percent']:.2f}%)",
                f"  Unseen occurrences: {occurrences['unseen_occurrences']} "
                f"({occurrences['unseen_percent']:.2f}%)",
                f"  Unique evaluation observation+mask states: {report['unique_evaluation_observation_mask_hashes']}",
                f"  Reservoir-covered unique states: {unique['reservoir_covered_unique_states']} "
                f"({unique['reservoir_covered_percent']:.2f}%)",
                f"  Observation-seen/mask-unseen unique states: "
                f"{unique['observation_seen_mask_unseen_unique_states']} "
                f"({unique['observation_seen_mask_unseen_percent']:.2f}%)",
            ))
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write("\n".join(lines) + "\n")
        return {"summary_json": json_path, "summary_text": text_path}


def _load_red_final_reservoir_coverage_index(audit_directory: str) -> Tuple[set[str], set[Tuple[str, str]], str]:
    """Load Red final-reservoir fingerprints emitted by StrategyCoverageAudit."""
    index_path = os.path.join(audit_directory, "strategy_coverage_final_reservoir_index.jsonl")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            "Strategy-evaluation coverage requires the final-reservoir fingerprint index "
            f"written by StrategyCoverageAudit: {index_path}. The older aggregate-only "
            "coverage summary cannot classify exact evaluation states; rerun the selected "
            "training configuration with --strategy-coverage-audit enabled."
        )
    observation_hashes: set[str] = set()
    observation_mask_hashes: set[Tuple[str, str]] = set()
    with open(index_path, "r", encoding="utf-8") as index_file:
        for line_number, line in enumerate(index_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid training coverage index JSON at line {line_number}.") from error
            if record.get("player") != "red":
                continue
            observation_hash = record.get("observation_hash")
            mask_hash = record.get("action_mask_hash")
            if not isinstance(observation_hash, str) or not isinstance(mask_hash, str):
                raise ValueError(f"Invalid Red coverage index record at line {line_number}.")
            observation_hashes.add(observation_hash)
            observation_mask_hashes.add((observation_hash, mask_hash))
    if not observation_mask_hashes:
        raise ValueError("The training coverage index contains no Red final-reservoir samples.")
    return observation_hashes, observation_mask_hashes, index_path


def print_hardware_info() -> torch.device:
    """Report the exact runtime device selected for this experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("                    DEEP CFR HARDWARE DETECTION")
    print("=" * 70)
    print(f"PyTorch version:   {torch.__version__}")
    print(f"CUDA available:    {torch.cuda.is_available()}")
    print(f"CUDA version:      {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU name:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (CPU only)'}")
    print(f"Selected device:   {device}")
    print("=" * 70)
    return device


def _action_group(action_index: int) -> str:
    """Return the Phase 2 action family for one validated action index."""
    if action_index == 0:
        return "end_turn"
    if 1 <= action_index <= 2:
        return "attack"
    if action_index == 3:
        return "retreat"
    if 4 <= action_index <= 19:
        return "play_pokemon"
    if 20 <= action_index <= 43:
        return "attach_energy"
    if 44 <= action_index <= 67:
        return "play_potion"
    if 68 <= action_index <= 69:
        return "promote"
    raise ValueError(f"Action index {action_index} is outside the Phase 2 action space.")


def _new_action_diagnostics() -> Dict[str, Any]:
    """Create raw action and terminal-outcome counters for one evaluation."""
    return {
        "total_action_selections": 0,
        "total_legal_actions": 0,
        "action_zero_selections": 0,
        "action_counts": {group: 0 for group in _ACTION_GROUPS},
        "action_68_selections": 0,
        "action_69_selections": 0,
        "deck_out_losses": 0,
        "two_knockout_losses": 0,
        "bench_out_losses": 0,
        "unclassified_terminal_losses": 0,
        "max_step_nonterminal_games": 0,
    }


def _record_action_selection(
    diagnostics: Dict[str, Any], action_index: int, legal_action_count: int
) -> None:
    """Record one already-validated Red action without changing its policy."""
    if not 0 <= action_index < ACTION_DIM:
        raise ValueError(f"Evaluation selected out-of-range action {action_index}.")
    diagnostics["total_action_selections"] += 1
    diagnostics["total_legal_actions"] += legal_action_count
    diagnostics["action_counts"][_action_group(action_index)] += 1
    if action_index == 0:
        diagnostics["action_zero_selections"] += 1
    elif action_index == 68:
        diagnostics["action_68_selections"] += 1
    elif action_index == 69:
        diagnostics["action_69_selections"] += 1


def _record_red_terminal_outcome(diagnostics: Dict[str, Any], env: PokemonTCGEnv) -> None:
    """Infer a Red loss category from the completed Phase 2 game state.

    The environment does not expose a terminal-reason field.  Under the Phase
    2 engine, a Red Active of ``None`` at a Blue win means a knockout terminal:
    Blue's second knockout is a two-KO loss; otherwise Red had no Bench and it
    is a bench-out loss.  A remaining Red Active with an empty deck is a
    deck-out loss.  The final category retains any future terminal reason that
    cannot be distinguished from the current state alone.
    """
    state = env.state
    if state.winner_idx is None:
        diagnostics["max_step_nonterminal_games"] += 1
        return
    if state.winner_idx != 1:
        return

    red_player = state.players[0]
    if red_player.active is None:
        if state.knockout_counts[1] >= 2:
            diagnostics["two_knockout_losses"] += 1
        else:
            diagnostics["bench_out_losses"] += 1
    elif red_player.deck.remaining_count() == 0:
        diagnostics["deck_out_losses"] += 1
    else:
        diagnostics["unclassified_terminal_losses"] += 1


def _finalize_action_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Add safe frequencies while retaining the raw counters in the result."""
    total = diagnostics["total_action_selections"]
    denominator = float(total) if total else 1.0
    finalized = dict(diagnostics)
    finalized["action_counts"] = dict(diagnostics["action_counts"])
    finalized["action_frequencies"] = {
        group: count / denominator
        for group, count in diagnostics["action_counts"].items()
    }
    finalized["action_zero_frequency"] = diagnostics["action_zero_selections"] / denominator
    finalized["action_68_frequency"] = diagnostics["action_68_selections"] / denominator
    finalized["action_69_frequency"] = diagnostics["action_69_selections"] / denominator
    finalized["average_legal_actions_per_decision"] = (
        diagnostics["total_legal_actions"] / denominator
    )
    return finalized


def _sample_masked_strategy(
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    legal_actions: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Sample a strategy strictly over the current legal Phase 2 actions.

    ``StrategyNetwork.get_strategy`` already masks and normalizes its output.
    This defensive check makes the diagnostic evaluator's contract explicit:
    every illegal action has exactly zero probability, then legal mass alone is
    normalized before sampling.  Forced-promotion masks therefore restrict
    sampling to action 68 and/or 69 automatically.
    """
    probabilities = np.asarray(strategy, dtype=np.float64)
    mask = np.asarray(legal_mask, dtype=np.float64)
    if probabilities.shape != (ACTION_DIM,) or mask.shape != (ACTION_DIM,):
        raise ValueError("Strategy and legal mask must each have one value per Phase 2 action.")
    if legal_actions.size == 0:
        raise RuntimeError("Cannot sample from an empty legal-action set.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError("Strategy network returned invalid action probabilities.")
    if np.any(probabilities[mask == 0.0] != 0.0):
        raise RuntimeError("Strategy assigned non-zero probability to an illegal action.")

    legal_probabilities = probabilities[legal_actions]
    legal_mass = float(legal_probabilities.sum())
    if not np.isfinite(legal_mass) or legal_mass <= 0.0:
        raise RuntimeError("Strategy assigned no usable probability mass to legal actions.")
    legal_probabilities = legal_probabilities / legal_mass
    action_index = int(rng.choice(legal_actions, p=legal_probabilities))
    if not 0 <= action_index < ACTION_DIM or mask[action_index] == 0.0:
        raise RuntimeError("Masked strategy sampling selected an illegal action.")
    return action_index


def _regret_matching_distribution(
    regret_network: RegretNetwork,
    observation: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    """Apply the traversal's exact regret-matching rule to current network output.

    The regret network predicts one raw cumulative-regret value for every Phase
    2 action.  This diagnostic deliberately delegates positive-regret clipping,
    legal masking, normalization, and the uniform legal fallback to the shared
    ``regret_matching`` function used by External-Sampling MCCFR traversal.
    """
    observation_t = torch.as_tensor(
        np.asarray(observation, dtype=np.float32),
        dtype=torch.float32,
        device=regret_network.device,
    )
    with torch.no_grad():
        regrets = regret_network(observation_t).detach().cpu().numpy()
    return regret_matching(regrets, legal_mask)


def evaluate_against_random_blue(
    strategy_network: StrategyNetwork,
    device: torch.device,
    *,
    random_red_baseline: bool = False,
    policy_mode: str | None = None,
    regret_network: RegretNetwork | None = None,
    seed_ranges: Iterable[Tuple[int, int]] = SEED_RANGES,
    max_steps: int = 200,
    strategy_evaluation_coverage_audit: StrategyEvaluationCoverageAudit | None = None,
) -> Dict[str, Any]:
    """Evaluate a Red policy against the established Random Blue opponent.

    Every game uses the same seed ranges as DQN/PPO/LSTM-PPO.  The random Red
    baseline has its own NumPy generator so its moves do not perturb the Python
    ``random`` stream used by the environment's Blue policy.  ``policy_mode``
    may be ``"random"``, ``"greedy"`` (the former diagnostic argmax mode),
    ``"sampled"`` (sample the masked Deep CFR strategy), or
    ``"regret_matching"`` (sample the final RegretNetwork using traversal's
    exact regret-matching rule).  Omitting it preserves the prior
    ``random_red_baseline`` call interface.
    """
    if policy_mode is None:
        policy_mode = "random" if random_red_baseline else "greedy"
    if policy_mode not in _EVALUATION_POLICIES:
        raise ValueError(f"Unknown evaluation policy {policy_mode!r}.")
    if strategy_evaluation_coverage_audit is not None and policy_mode not in {"greedy", "sampled"}:
        raise ValueError("Strategy evaluation coverage is defined only for greedy and sampled StrategyNetwork policies.")

    env = PokemonTCGEnv()
    strategy_network.to(device)
    strategy_network.device = device
    strategy_network.eval()
    if policy_mode == "regret_matching":
        if regret_network is None:
            raise ValueError("Regret-matching evaluation requires a RegretNetwork.")
        regret_network.to(device)
        regret_network.device = device
        regret_network.eval()
    result: Dict[str, Any] = {
        "ranges": {},
        "overall": {
            "games": 0, "wins": 0, "losses": 0, "total_reward": 0.0,
            "total_steps": 0, "win_rate": 0.0, "avg_reward": 0.0, "avg_length": 0.0,
        },
        "action_diagnostics": _new_action_diagnostics(),
    }

    for start_seed, end_seed in seed_ranges:
        wins = 0
        total_reward = 0.0
        total_steps = 0
        games = end_seed - start_seed + 1
        range_diagnostics = _new_action_diagnostics()
        for seed in range(start_seed, end_seed + 1):
            observation, _ = env.reset(seed=seed)
            # Independent generator keeps the Random Blue action stream fair.
            red_rng = np.random.default_rng(seed + 1_000_000)
            terminated = truncated = False
            reward_sum = 0.0
            steps = 0
            while not (terminated or truncated) and steps < max_steps:
                legal_mask = env.get_action_mask()
                legal_actions = np.flatnonzero(legal_mask)
                if legal_actions.size == 0:
                    raise RuntimeError("Red received no legal action during evaluation.")
                if policy_mode == "random":
                    action_index = int(red_rng.choice(legal_actions))
                elif policy_mode in {"greedy", "sampled"}:
                    strategy = strategy_network.get_strategy(observation, legal_mask)
                    if policy_mode == "greedy":
                        action_index = int(np.argmax(strategy))
                    else:
                        action_index = _sample_masked_strategy(
                            strategy, legal_mask, legal_actions, red_rng
                        )
                else:
                    regret_strategy = _regret_matching_distribution(
                        regret_network, observation, legal_mask
                    )
                    action_index = _sample_masked_strategy(
                        regret_strategy, legal_mask, legal_actions, red_rng
                    )
                if not 0 <= action_index < ACTION_DIM or legal_mask[action_index] == 0:
                    raise RuntimeError("Evaluation policy selected an illegal action.")
                if strategy_evaluation_coverage_audit is not None:
                    state = env.state
                    strategy_evaluation_coverage_audit.record_decision(
                        policy_mode=policy_mode,
                        observation=observation,
                        action_mask=legal_mask,
                        game_step=steps,
                        turn_count=int(state.turn_count) if state is not None else 0,
                        selected_action=action_index,
                        selected_action_category=_action_group(action_index),
                        forced_promotion=(
                            state is not None and state.forced_promotion_player_idx == 0
                        ),
                    )
                _record_action_selection(range_diagnostics, action_index, legal_actions.size)
                _record_action_selection(
                    result["action_diagnostics"], action_index, legal_actions.size
                )
                observation, reward, terminated, truncated, _ = env.step(action_index)
                reward_sum += float(reward)
                steps += 1
            if env.state.winner_idx == 0:
                wins += 1
            total_reward += reward_sum
            total_steps += steps
            _record_red_terminal_outcome(range_diagnostics, env)
            _record_red_terminal_outcome(result["action_diagnostics"], env)

        losses = games - wins
        label = f"{start_seed}-{end_seed}"
        result["ranges"][label] = {
            "games": games,
            "wins": wins,
            "losses": losses,
            "win_rate": 100.0 * wins / games,
            "avg_reward": total_reward / games,
            "avg_length": total_steps / games,
            "action_diagnostics": _finalize_action_diagnostics(range_diagnostics),
        }
        overall = result["overall"]
        overall["games"] += games
        overall["wins"] += wins
        overall["losses"] += losses
        overall["total_reward"] += total_reward
        overall["total_steps"] += total_steps

    overall = result["overall"]
    overall["win_rate"] = 100.0 * overall["wins"] / overall["games"]
    overall["avg_reward"] = overall["total_reward"] / overall["games"]
    overall["avg_length"] = overall["total_steps"] / overall["games"]
    result["action_diagnostics"] = _finalize_action_diagnostics(result["action_diagnostics"])
    return result


def generate_performance_plots(
    history: Dict[str, List[Any]],
    random_results: Dict[str, Any],
    greedy_results: Dict[str, Any],
    sampled_results: Dict[str, Any],
    regret_matching_results: Dict[str, Any],
    output_path: str,
) -> None:
    """Save the requested training and benchmark dashboard."""
    iterations = history["iteration"]
    figure, axes = plt.subplots(3, 3, figsize=(18, 14))
    figure.suptitle("Deep CFR Training and Evaluation — Mini Pokemon TCG", fontsize=16, fontweight="bold")

    axes[0, 0].plot(iterations, history["red_traversal_utility"], label="Red traverser")
    axes[0, 0].plot(iterations, history["blue_traversal_utility"], label="Blue traverser")
    axes[0, 0].set(title="Traversal utility", xlabel="Outer iteration", ylabel="Utility")
    axes[0, 0].legend()

    axes[0, 1].plot(iterations, history["red_regret_network_loss"], label="Red")
    axes[0, 1].plot(iterations, history["blue_regret_network_loss"], label="Blue")
    axes[0, 1].set(title="Regret-network MSE", xlabel="Outer iteration", ylabel="Loss")
    axes[0, 1].legend()

    axes[0, 2].plot(iterations, history["red_strategy_network_loss"], label="Red")
    axes[0, 2].plot(iterations, history["blue_strategy_network_loss"], label="Blue")
    axes[0, 2].set(title="Masked strategy cross-entropy", xlabel="Outer iteration", ylabel="Loss")
    axes[0, 2].legend()

    axes[1, 0].plot(iterations, history["red_regret_reservoir_size"], label="Red regret")
    axes[1, 0].plot(iterations, history["blue_regret_reservoir_size"], label="Blue regret")
    axes[1, 0].set(title="Regret reservoir growth", xlabel="Outer iteration", ylabel="Samples")
    axes[1, 0].legend()

    axes[1, 1].plot(iterations, history["red_strategy_reservoir_size"], label="Red strategy")
    axes[1, 1].plot(iterations, history["blue_strategy_reservoir_size"], label="Blue strategy")
    axes[1, 1].set(title="Strategy reservoir growth", xlabel="Outer iteration", ylabel="Samples")
    axes[1, 1].legend()

    axes[1, 2].plot(iterations, history["elapsed_training_time"], color="tab:purple")
    axes[1, 2].set(title="Training progress", xlabel="Outer iteration", ylabel="Seconds / iteration")

    names = [
        "Random\nBaseline",
        "Deep CFR\nGreedy",
        "Deep CFR\nSampled",
        "Deep CFR\nRegret-Matching",
    ]
    summaries = [
        random_results["overall"],
        greedy_results["overall"],
        sampled_results["overall"],
        regret_matching_results["overall"],
    ]
    colors = ["salmon", "royalblue", "seagreen", "darkorchid"]
    axes[2, 0].bar(names, [item["win_rate"] for item in summaries], color=colors)
    axes[2, 0].set(title="Evaluation win rate", ylabel="Win rate (%)", ylim=(0, 100))

    axes[2, 1].bar(names, [item["avg_length"] for item in summaries], color=colors)
    axes[2, 1].set(title="Evaluation average game length", ylabel="Steps")

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.9,
        "Reference benchmarks\n\nBest DQN = 99.3%\nFinal PPO = 99.5%\nFinal LSTM-PPO = 99.5%\n\n"
        "Network losses measure fitting,\nnot playing strength.",
        va="top",
        fontsize=12,
    )
    for axis in axes.flat:
        axis.grid(axis="y", linestyle=":", alpha=0.45)
    axes[2, 2].grid(False)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _print_table(rows: List[Tuple[str, Dict[str, Any]]]) -> None:
    print("\n" + "=" * 96)
    print("FINAL COMPARATIVE BENCHMARK TABLE")
    print("=" * 96)
    print(f"{'Model':20} | {'Games':5} | {'Wins':5} | {'Losses':6} | {'Win Rate':8} | {'Avg Reward':10} | {'Avg Length':10}")
    print("-" * 96)
    for name, result in rows:
        item = result["overall"]
        print(
            f"{name:20} | {item['games']:5d} | {item['wins']:5d} | {item['losses']:6d} | "
            f"{item['win_rate']:7.1f}% | {item['avg_reward']:10.2f} | {item['avg_length']:10.1f}"
        )
    print("=" * 96)
    print("Reference: Best DQN = 99.3% | Final PPO = 99.5% | Final LSTM-PPO = 99.5%")
    print("Deep CFR playing strength is determined by this evaluation, not training loss.")


def _print_seed_range_breakdown(rows: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Match the per-seed-range evaluation report used by PPO/LSTM-PPO."""
    print("\n" + "=" * 80)
    print("SEED RANGE PERFORMANCE BREAKDOWN")
    print("=" * 80)
    for model_name, result in rows:
        print(f"\nModel: {model_name}")
        print("-" * 50)
        for range_label, data in result["ranges"].items():
            print(
                f"  Range {range_label} | Games: {data['games']} | Wins: {data['wins']} | "
                f"Losses: {data['losses']} | Win Rate: {data['win_rate']:.1f}% | "
                f"Avg Reward: {data['avg_reward']:.2f} | Avg Length: {data['avg_length']:.1f}"
            )


def _print_action_diagnostics(model_name: str, result: Dict[str, Any]) -> None:
    """Print the requested policy-behaviour diagnostic for one evaluation."""
    diagnostics = result["action_diagnostics"]
    print(f"\n{model_name} action diagnostics")
    print("-" * 80)
    print(f"  Total action selections: {diagnostics['total_action_selections']}")
    print(
        "  Average legal actions / decision: "
        f"{diagnostics['average_legal_actions_per_decision']:.2f}"
    )
    print(
        f"  Action 0 (end_turn): {diagnostics['action_zero_selections']} "
        f"({100.0 * diagnostics['action_zero_frequency']:.2f}%)"
    )
    for group in _ACTION_GROUPS:
        print(
            f"  {group}: {diagnostics['action_counts'][group]} "
            f"({100.0 * diagnostics['action_frequencies'][group]:.2f}%)"
        )
    print(
        f"  Forced promotion action 68: {diagnostics['action_68_selections']} "
        f"({100.0 * diagnostics['action_68_frequency']:.2f}%)"
    )
    print(
        f"  Forced promotion action 69: {diagnostics['action_69_selections']} "
        f"({100.0 * diagnostics['action_69_frequency']:.2f}%)"
    )
    print(f"  Deck-out losses: {diagnostics['deck_out_losses']}")
    print(f"  Two-knockout losses: {diagnostics['two_knockout_losses']}")
    print(f"  Bench-out losses: {diagnostics['bench_out_losses']}")
    print(f"  Unclassified terminal losses: {diagnostics['unclassified_terminal_losses']}")
    print(f"  Nonterminal games at max steps: {diagnostics['max_step_nonterminal_games']}")


def _print_final_summary(
    device: torch.device,
    training_time: float | None,
    random_results: Dict[str, Any],
    greedy_results: Dict[str, Any],
    sampled_results: Dict[str, Any],
    regret_matching_results: Dict[str, Any],
    metrics_path: str,
    plot_path: str | None,
) -> None:
    print("\n" + "=" * 70)
    print("TCG EXPERIMENT FINAL SUMMARY")
    print("=" * 70)
    print(f"Device used:                    {device}")
    if training_time is not None:
        print(f"Total training time:            {training_time:.1f} seconds")
    else:
        print("Total training time:            not run (evaluation-only)")
    print(f"Random baseline:                {random_results['overall']['win_rate']:.1f}%")
    print(f"Deep CFR (greedy strategy):     {greedy_results['overall']['win_rate']:.1f}%")
    print(f"Deep CFR (sampled strategy):    {sampled_results['overall']['win_rate']:.1f}%")
    print(
        "Deep CFR (regret matching):    "
        f"{regret_matching_results['overall']['win_rate']:.1f}%"
    )
    print("Best DQN:                       99.3% (benchmark reference)")
    print("Final PPO:                      99.5% (benchmark reference)")
    print("Final LSTM-PPO:                 99.5% (benchmark reference)")
    if training_time is not None:
        print(f"Metrics saved to:               {metrics_path}")
    else:
        print(f"Training metrics (unchanged):   {metrics_path}")
    print(f"Plots saved to:                 {plot_path if plot_path is not None else 'not generated (evaluation-only)'}")
    print("=" * 70)


def _print_horizon_pilot_summary(
    history: Dict[str, List[Any]],
    trainer: Any,
    training_time: float,
    results_dir: str,
    *,
    title: str = "DEEP CFR HORIZON-8 PILOT SUMMARY",
    explicit_horizon: int = PILOT_MAX_STEPS,
) -> None:
    """Report the requested compact pilot diagnostics without evaluation."""
    red_records = [
        record for record in trainer.traversal_records if record["traverser"] == 0
    ]
    blue_records = [
        record for record in trainer.traversal_records if record["traverser"] == 1
    ]

    def traversal_time(records: List[Dict[str, Any]]) -> Tuple[float, float]:
        total = sum(float(record["diagnostics"]["elapsed_seconds"]) for record in records)
        return total, total / len(records) if records else 0.0

    red_total, red_average = traversal_time(red_records)
    blue_total, blue_average = traversal_time(blue_records)
    print("\n" + "=" * 70)
    print(f"               {title}")
    print("=" * 70)
    print("Iterations / traversals per player: 10 / 16")
    print(f"Explicit MCCFR horizon:            {explicit_horizon}")
    print(f"Final Red / Blue utility:          {history['red_traversal_utility'][-1]:+.3f} / {history['blue_traversal_utility'][-1]:+.3f}")
    print(f"Final Red / Blue regret loss:      {history['red_regret_network_loss'][-1]:.6f} / {history['blue_regret_network_loss'][-1]:.6f}")
    print(f"Final Red / Blue strategy loss:    {history['red_strategy_network_loss'][-1]:.6f} / {history['blue_strategy_network_loss'][-1]:.6f}")
    print(f"Red traversal time total / mean:   {red_total:.1f}s / {red_average:.2f}s")
    print(f"Blue traversal time total / mean:  {blue_total:.1f}s / {blue_average:.2f}s")
    print(f"Total pilot training time:         {training_time:.1f}s")
    print(f"Pilot outputs:                     {results_dir}")
    print("Evaluation:                        skipped by pilot design")
    print("=" * 70)


def _print_horizon6_runtime_summary(
    history: Dict[str, List[Any]],
    trainer: Any,
    training_time: float,
    results_dir: str,
) -> None:
    """Report one full horizon-6 iteration's requested runtime counters."""
    records = trainer.traversal_records
    red_records = [record for record in records if record["traverser"] == 0]
    blue_records = [record for record in records if record["traverser"] == 1]

    def traversal_time(records_for_player: List[Dict[str, Any]]) -> Tuple[float, float]:
        total = sum(
            float(record["diagnostics"]["elapsed_seconds"])
            for record in records_for_player
        )
        return total, total / len(records_for_player) if records_for_player else 0.0

    def total_diagnostic(key: str) -> int:
        return sum(int(record["diagnostics"][key]) for record in records)

    red_total, red_average = traversal_time(red_records)
    blue_total, blue_average = traversal_time(blue_records)
    print("\n" + "=" * 70)
    print("          DEEP CFR HORIZON-6 ONE-ITERATION RUNTIME TEST")
    print("=" * 70)
    print("Iterations / traversals per player: 1 / 128")
    print("Explicit MCCFR horizon:            6")
    print(f"Total runtime:                     {training_time:.1f}s")
    print(f"Red traversal time total / mean:   {red_total:.1f}s / {red_average:.2f}s")
    print(f"Blue traversal time total / mean:  {blue_total:.1f}s / {blue_average:.2f}s")
    print(f"Total decision nodes:              {total_diagnostic('decision_nodes')}")
    print(f"Total continuation rollouts:       {total_diagnostic('horizon_rollouts')}")
    print(f"Total snapshots / deep copies:     {total_diagnostic('state_copies')}")
    print(f"Total GameEngine transitions:      {total_diagnostic('game_transitions')}")
    print(f"Red / Blue regret loss:            {history['red_regret_network_loss'][-1]:.6f} / {history['blue_regret_network_loss'][-1]:.6f}")
    print(f"Red / Blue strategy loss:          {history['red_strategy_network_loss'][-1]:.6f} / {history['blue_strategy_network_loss'][-1]:.6f}")
    print(f"Runtime-test outputs:              {results_dir}")
    print("Evaluation:                        skipped by runtime-test design")
    print("=" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse runner controls without changing the experiment configuration."""
    parser = argparse.ArgumentParser(description="Train or diagnostically evaluate Deep CFR.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load the existing final Red strategy and regret checkpoints and skip training.",
    )
    mode_group.add_argument(
        "--pilot",
        action="store_true",
        help="Run the separate 10-iteration, horizon-8 Deep CFR pilot without evaluation.",
    )
    mode_group.add_argument(
        "--iteration-weighted-depth4-pilot",
        action="store_true",
        help=(
            "Run the separate 10-iteration iteration-weighted StrategyNetwork, "
            "horizon-4 Deep CFR pilot without evaluation."
        ),
    )
    mode_group.add_argument(
        "--iteration-weighted-depth6-pilot",
        action="store_true",
        help=(
            "Run the separate 10-iteration iteration-weighted StrategyNetwork, "
            "horizon-6 Deep CFR pilot without evaluation."
        ),
    )
    mode_group.add_argument(
        "--horizon6-runtime-test",
        action="store_true",
        help="Run one horizon-6 iteration with 128 Red and 128 Blue traversals, then skip evaluation.",
    )
    mode_group.add_argument(
        "--strategy-evaluation-coverage",
        action="store_true",
        help=(
            "Run the read-only 200-game greedy-versus-sampled StrategyNetwork coverage "
            "diagnostic for an explicitly supplied checkpoint."
        ),
    )
    parser.add_argument(
        "--strategy-coverage-audit",
        action="store_true",
        help=(
            "Passively write StrategyNetwork reservoir-coverage diagnostics without "
            "changing Deep CFR decisions, training, or evaluation."
        ),
    )
    parser.add_argument(
        "--strategy-checkpoint",
        type=str,
        help="Explicit StrategyNetwork checkpoint path for --strategy-evaluation-coverage.",
    )
    parser.add_argument(
        "--training-coverage-audit-dir",
        type=str,
        help=(
            "Directory containing strategy_coverage_final_reservoir_index.jsonl for "
            "--strategy-evaluation-coverage."
        ),
    )
    return parser.parse_args()


def _load_final_red_strategy(results_dir: str, device: torch.device) -> StrategyNetwork:
    """Load the fresh iteration-weighted masked-CE Phase 2 Red strategy checkpoint."""
    final_strategy_path = os.path.join(
        results_dir, "final_deep_cfr_red_strategy_masked_ce_iteration_weighted.pt"
    )
    if not os.path.isfile(final_strategy_path):
        raise FileNotFoundError(
            "Evaluation requires a freshly trained iteration-weighted masked-cross-entropy "
            "Deep CFR Red strategy checkpoint. Earlier unweighted and legacy-MSE strategy "
            f"checkpoints are intentionally not used: {final_strategy_path}"
        )
    red_strategy = StrategyNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    red_strategy.device = device
    red_strategy.load_state_dict(torch.load(final_strategy_path, map_location=device))
    return red_strategy


def _load_strategy_checkpoint(strategy_checkpoint: str, device: torch.device) -> StrategyNetwork:
    """Load one explicit Phase 2 StrategyNetwork checkpoint for read-only diagnostics."""
    checkpoint_path = os.path.abspath(strategy_checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Strategy evaluation coverage checkpoint was not found: {checkpoint_path}")
    strategy = StrategyNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    strategy.device = device
    try:
        strategy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise RuntimeError(
            "Could not load the supplied Phase 2 StrategyNetwork checkpoint for "
            f"evaluation coverage: {checkpoint_path}"
        ) from error
    return strategy


def _strategy_evaluation_coverage_output_directory(strategy_checkpoint: str) -> str:
    """Return a checkpoint-specific directory without touching training outputs."""
    checkpoint_path = os.path.abspath(strategy_checkpoint)
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    path_hash = hashlib.blake2b(checkpoint_path.encode("utf-8"), digest_size=6).hexdigest()
    return os.path.join(
        SCRIPT_DIR,
        "results",
        STRATEGY_EVALUATION_COVERAGE_RESULTS_DIRECTORY,
        f"{checkpoint_stem}_{path_hash}",
    )


def _run_strategy_evaluation_coverage(
    *,
    strategy_checkpoint: str,
    training_coverage_audit_dir: str,
    device: torch.device,
) -> None:
    """Run the opt-in 200-game read-only greedy/sampled coverage comparison."""
    training_observations, training_observation_masks, index_path = (
        _load_red_final_reservoir_coverage_index(training_coverage_audit_dir)
    )
    strategy = _load_strategy_checkpoint(strategy_checkpoint, device)
    coverage_audit = StrategyEvaluationCoverageAudit(
        training_observation_hashes=training_observations,
        training_observation_mask_hashes=training_observation_masks,
        training_coverage_index_path=index_path,
    )
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for policy_mode, label in (
        ("greedy", "Strategy Greedy"),
        ("sampled", "Strategy Sampled"),
    ):
        result = evaluate_against_random_blue(
            strategy,
            device,
            policy_mode=policy_mode,
            seed_ranges=STRATEGY_EVALUATION_COVERAGE_SEED_RANGES,
            max_steps=STRATEGY_EVALUATION_COVERAGE_MAX_STEPS,
            strategy_evaluation_coverage_audit=coverage_audit,
        )
        rows.append((label, result))

    output_directory = _strategy_evaluation_coverage_output_directory(strategy_checkpoint)
    output_paths = coverage_audit.write_report(output_directory)
    _print_seed_range_breakdown(rows)
    _print_table(rows)
    for model_name, result in rows:
        _print_action_diagnostics(model_name, result)
    print("\nStrategy evaluation coverage reports:")
    print(f"  JSON: {output_paths['summary_json']}")
    print(f"  Text: {output_paths['summary_text']}")


def _load_final_red_regret(results_dir: str, device: torch.device) -> RegretNetwork:
    """Load the final Phase 2 Red regret checkpoint for diagnostic evaluation."""
    final_regret_path = os.path.join(results_dir, "final_deep_cfr_red_regret.pt")
    if not os.path.isfile(final_regret_path):
        raise FileNotFoundError(
            "Regret-matching evaluation requires the existing final Deep CFR Red "
            f"regret checkpoint: {final_regret_path}"
        )
    red_regret = RegretNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    red_regret.device = device
    try:
        red_regret.load_state_dict(torch.load(final_regret_path, map_location=device))
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise RuntimeError(
            "Could not load the final Deep CFR Red regret checkpoint for "
            f"regret-matching evaluation: {final_regret_path}"
        ) from error
    return red_regret


def main(
    *,
    evaluate_only: bool = False,
    pilot: bool = False,
    iteration_weighted_depth4_pilot: bool = False,
    iteration_weighted_depth6_pilot: bool = False,
    horizon6_runtime_test: bool = False,
    strategy_coverage_audit: bool = False,
    strategy_evaluation_coverage: bool = False,
    strategy_checkpoint: str | None = None,
    training_coverage_audit_dir: str | None = None,
) -> None:
    device = print_hardware_info()
    if strategy_evaluation_coverage:
        if not strategy_checkpoint or not training_coverage_audit_dir:
            raise ValueError(
                "--strategy-evaluation-coverage requires both --strategy-checkpoint and "
                "--training-coverage-audit-dir."
            )
        _run_strategy_evaluation_coverage(
            strategy_checkpoint=strategy_checkpoint,
            training_coverage_audit_dir=training_coverage_audit_dir,
            device=device,
        )
        return
    if horizon6_runtime_test:
        results_dir = os.path.join(SCRIPT_DIR, "results", RUNTIME_TEST_RESULTS_DIRECTORY)
    elif iteration_weighted_depth6_pilot:
        results_dir = os.path.join(
            SCRIPT_DIR, "results", ITERATION_WEIGHTED_DEPTH6_PILOT_RESULTS_DIRECTORY
        )
    elif iteration_weighted_depth4_pilot:
        results_dir = os.path.join(
            SCRIPT_DIR, "results", ITERATION_WEIGHTED_PILOT_RESULTS_DIRECTORY
        )
    elif pilot:
        results_dir = os.path.join(SCRIPT_DIR, "results", PILOT_RESULTS_DIRECTORY)
    else:
        results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    history: Dict[str, List[Any]] | None = None
    training_time: float | None = None
    if evaluate_only:
        print("\nEvaluation-only mode: loading the existing final Phase 2 strategy and regret checkpoints.")
        red_strategy = _load_final_red_strategy(results_dir, device)
        red_regret = _load_final_red_regret(results_dir, device)
    else:
        # Passing this explicitly ensures the real experiment runs on CUDA whenever
        # it is available; the trainer moves all four networks and training tensors
        # to this same device. CPU remains the safe fallback for machines without it.
        config = DeepCFRConfig(
            num_iterations=(
                RUNTIME_TEST_NUM_ITERATIONS if horizon6_runtime_test
                else ITERATION_WEIGHTED_DEPTH6_PILOT_NUM_ITERATIONS if iteration_weighted_depth6_pilot
                else ITERATION_WEIGHTED_PILOT_NUM_ITERATIONS if iteration_weighted_depth4_pilot
                else PILOT_NUM_ITERATIONS if pilot else NUM_ITERATIONS
            ),
            traversals_per_iter=(
                RUNTIME_TEST_TRAVERSALS_PER_ITER if horizon6_runtime_test
                else ITERATION_WEIGHTED_DEPTH6_PILOT_TRAVERSALS_PER_ITER if iteration_weighted_depth6_pilot
                else ITERATION_WEIGHTED_PILOT_TRAVERSALS_PER_ITER if iteration_weighted_depth4_pilot
                else PILOT_TRAVERSALS_PER_ITER if pilot else TRAVERSALS_PER_ITER
            ),
            batch_size=BATCH_SIZE,
            reservoir_capacity=RESERVOIR_CAPACITY,
            learning_rate=LEARNING_RATE,
            regret_epochs=REGRET_EPOCHS,
            strategy_epochs=STRATEGY_EPOCHS,
            max_steps=(
                RUNTIME_TEST_MAX_STEPS if horizon6_runtime_test
                else ITERATION_WEIGHTED_DEPTH6_PILOT_MAX_STEPS if iteration_weighted_depth6_pilot
                else ITERATION_WEIGHTED_PILOT_MAX_STEPS if iteration_weighted_depth4_pilot
                else PILOT_MAX_STEPS if pilot else MAX_STEPS
            ),
            checkpoint_dir=results_dir,
            seed=BASE_SEED,
            device=device,
            strategy_coverage_audit=strategy_coverage_audit,
        )
        heading = (
            "\nDeep CFR horizon-6 runtime-test configuration:"
            if horizon6_runtime_test
            else "\nDeep CFR iteration-weighted depth-6 pilot configuration:"
            if iteration_weighted_depth6_pilot
            else "\nDeep CFR iteration-weighted depth-4 pilot configuration:"
            if iteration_weighted_depth4_pilot
            else "\nDeep CFR horizon-8 pilot configuration:"
            if pilot
            else "\nDeep CFR experiment configuration:"
        )
        print(heading)
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")

        print(f"\nTraining Deep CFR on {device} for {config.num_iterations} iterations...")
        training_start = time.time()
        trainer, history = train_deep_cfr(config, verbose=True, show_hardware=False)
        training_time = time.time() - training_start
        print(f"\nTraining complete! Time taken: {training_time:.1f} seconds.")
        if pilot:
            _print_horizon_pilot_summary(history, trainer, training_time, results_dir)
            return
        if iteration_weighted_depth4_pilot:
            _print_horizon_pilot_summary(
                history,
                trainer,
                training_time,
                results_dir,
                title="DEEP CFR ITERATION-WEIGHTED DEPTH-4 PILOT SUMMARY",
                explicit_horizon=ITERATION_WEIGHTED_PILOT_MAX_STEPS,
            )
            return
        if iteration_weighted_depth6_pilot:
            _print_horizon_pilot_summary(
                history,
                trainer,
                training_time,
                results_dir,
                title="DEEP CFR ITERATION-WEIGHTED DEPTH-6 PILOT SUMMARY",
                explicit_horizon=ITERATION_WEIGHTED_DEPTH6_PILOT_MAX_STEPS,
            )
            return
        if horizon6_runtime_test:
            _print_horizon6_runtime_summary(history, trainer, training_time, results_dir)
            return
        red_strategy = _load_final_red_strategy(results_dir, device)
        red_regret = _load_final_red_regret(results_dir, device)

    print("\nEvaluating random Red baseline against the established Random Blue opponent (1,500 games)...")
    random_results = evaluate_against_random_blue(red_strategy, device, policy_mode="random")
    print("Evaluating greedy Deep CFR Red against the established Random Blue opponent (1,500 games)...")
    greedy_results = evaluate_against_random_blue(red_strategy, device, policy_mode="greedy")
    print("Evaluating sampled-strategy Deep CFR Red against the established Random Blue opponent (1,500 games)...")
    sampled_results = evaluate_against_random_blue(red_strategy, device, policy_mode="sampled")
    print("Evaluating Deep CFR Regret-Matching Red against the established Random Blue opponent (1,500 games)...")
    regret_matching_results = evaluate_against_random_blue(
        red_strategy,
        device,
        policy_mode="regret_matching",
        regret_network=red_regret,
    )

    plot_path: str | None = None
    if history is not None:
        print("\nGenerating Deep CFR performance plots...")
        plot_path = os.path.join(results_dir, "deep_cfr_performance_plots.png")
        generate_performance_plots(
            history,
            random_results,
            greedy_results,
            sampled_results,
            regret_matching_results,
            plot_path,
        )
    metrics_path = os.path.join(results_dir, "deep_cfr_training_metrics.csv")
    models_data = [
        ("Random Baseline", random_results),
        ("Deep CFR Greedy", greedy_results),
        ("Deep CFR Sampled", sampled_results),
        ("Deep CFR Regret-Matching", regret_matching_results),
    ]
    _print_seed_range_breakdown(models_data)
    _print_table(models_data)
    for model_name, model_result in models_data:
        _print_action_diagnostics(model_name, model_result)
    _print_final_summary(
        device,
        training_time,
        random_results,
        greedy_results,
        sampled_results,
        regret_matching_results,
        metrics_path,
        plot_path,
    )


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        evaluate_only=arguments.evaluate_only,
        pilot=arguments.pilot,
        iteration_weighted_depth4_pilot=arguments.iteration_weighted_depth4_pilot,
        iteration_weighted_depth6_pilot=arguments.iteration_weighted_depth6_pilot,
        horizon6_runtime_test=arguments.horizon6_runtime_test,
        strategy_coverage_audit=arguments.strategy_coverage_audit,
        strategy_evaluation_coverage=arguments.strategy_evaluation_coverage,
        strategy_checkpoint=arguments.strategy_checkpoint,
        training_coverage_audit_dir=arguments.training_coverage_audit_dir,
    )
