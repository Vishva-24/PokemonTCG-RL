"""Two-agent cross-play and self-play tournament for the Phase 2 TCG.

The runner advances :class:`GameEngine` directly, routing only the current
player's 50-D observation and 70-D legal mask to that player's independent
adapter. Results are appended one completed game at a time for safe resume.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from pokemon_rl.evaluation.tournament_agents import (
    ALGORITHM_NAMES,
    DEFAULT_CHECKPOINTS,
    PROJECT_ROOT,
    AgentAdapter,
    default_checkpoint_path,
    load_agent,
)
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, PokemonTCGEnv, get_phase2_information_set_observation
from pokemon_rl.game import GameEngine, GameState


CROSS_MATCHUPS = (
    ("DQN", "PPO"),
    ("DQN", "LSTM-PPO"),
    ("DQN", "Deep CFR"),
    ("PPO", "LSTM-PPO"),
    ("PPO", "Deep CFR"),
    ("LSTM-PPO", "Deep CFR"),
)
SELF_PLAY_ALGORITHMS = ("DQN", "PPO", "LSTM-PPO", "Deep CFR")
RESULTS_ROOT = PROJECT_ROOT / "results" / "head_to_head_tournament"
GAME_RESULTS_FILENAME = "game_level_results.csv"
OUTPUT_FILENAMES = (
    "tournament_summary.json",
    "tournament_summary.csv",
    "game_level_results.csv",
    "pairwise_win_matrix.csv",
    "self_play_results.csv",
    "tournament_report.txt",
    "tournament_config.json",
)
Z_95 = 1.959963984540054

GAME_RESULT_FIELDS = (
    "match_type",
    "matchup",
    "orientation",
    "algorithm_red",
    "algorithm_blue",
    "red_instance",
    "blue_instance",
    "seed",
    "decisions",
    "terminal",
    "step_cap_reached",
    "terminal_reason",
    "winner",
    "reason_player",
    "red_reward",
    "blue_reward",
    "red_knockouts",
    "blue_knockouts",
    "illegal_action_attempts",
)

SUMMARY_FIELDS = (
    "match_type",
    "matchup",
    "orientation",
    "algorithm_red",
    "algorithm_blue",
    "games",
    "terminal_games",
    "red_wins",
    "red_losses",
    "red_win_rate_percent",
    "red_win_rate_ci95_low_percent",
    "red_win_rate_ci95_high_percent",
    "blue_wins",
    "blue_losses",
    "blue_win_rate_percent",
    "blue_win_rate_ci95_low_percent",
    "blue_win_rate_ci95_high_percent",
    "red_reward_average",
    "blue_reward_average",
    "average_decisions_per_game",
    "red_two_knockout_victories",
    "blue_two_knockout_victories",
    "red_deck_out_losses",
    "blue_deck_out_losses",
    "red_bench_out_losses",
    "blue_bench_out_losses",
    "nonterminal_at_step_cap",
    "unclassified_terminal_games",
    "illegal_action_attempts",
)


class IllegalActionError(RuntimeError):
    """An agent selected an action excluded by its current legal mask."""


class TournamentExecutionError(RuntimeError):
    """A tournament execution invariant failed; no fallback action is used."""


@dataclass(frozen=True)
class TournamentConfig:
    games_per_orientation: int = 200
    max_decisions: int = 200
    seed: int = 12345
    device: str = "cpu"
    mode: str = "all"

    def validate(self) -> None:
        if self.games_per_orientation <= 0:
            raise ValueError("games_per_orientation must be positive.")
        if self.max_decisions <= 0:
            raise ValueError("max_decisions must be positive.")
        if self.mode not in {"all", "cross_play_only", "self_play_only"}:
            raise ValueError(f"Unknown tournament mode {self.mode!r}.")
        try:
            torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid torch device {self.device!r}.") from exc

    @property
    def expected_games(self) -> int:
        orientations = 0
        if self.mode in {"all", "cross_play_only"}:
            orientations += len(CROSS_MATCHUPS) * 2
        if self.mode in {"all", "self_play_only"}:
            orientations += len(SELF_PLAY_ALGORITHMS) * 2
        return orientations * self.games_per_orientation


@dataclass(frozen=True)
class OrientationPlan:
    match_type: str
    matchup: str
    orientation: str
    algorithm_red: str
    algorithm_blue: str
    red_instance: int
    blue_instance: int

    @property
    def key(self) -> tuple[str, str, str]:
        return self.match_type, self.matchup, self.orientation


def build_orientation_plans(mode: str = "all") -> list[OrientationPlan]:
    if mode not in {"all", "cross_play_only", "self_play_only"}:
        raise ValueError(f"Unknown tournament mode {mode!r}.")
    plans: list[OrientationPlan] = []
    if mode in {"all", "cross_play_only"}:
        for first, second in CROSS_MATCHUPS:
            plans.append(OrientationPlan(
                "cross_play", f"{first} vs {second}", f"{first}_red__{second}_blue",
                first, second, 1, 2,
            ))
            plans.append(OrientationPlan(
                "cross_play", f"{first} vs {second}", f"{second}_red__{first}_blue",
                second, first, 2, 1,
            ))
    if mode in {"all", "self_play_only"}:
        for algorithm in SELF_PLAY_ALGORITHMS:
            plans.append(OrientationPlan(
                "self_play", f"{algorithm} vs {algorithm}", "instance_1_red__instance_2_blue",
                algorithm, algorithm, 1, 2,
            ))
            plans.append(OrientationPlan(
                "self_play", f"{algorithm} vs {algorithm}", "instance_2_red__instance_1_blue",
                algorithm, algorithm, 2, 1,
            ))
    return plans


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval as percentages."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials.")
    if trials == 0:
        return 0.0, 100.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))) / denom
    return 100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp"
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _atomic_write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, buffer.getvalue())


def create_unique_run_directory(
    base_directory: Path = RESULTS_ROOT, seed: int = 12345
) -> Path:
    base_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"run_{timestamp}_seed{seed}"
    candidate = base_directory / stem
    suffix = 1
    while candidate.exists():
        candidate = base_directory / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _seeded_initial_state(seed: int) -> GameState:
    """Seed only engine deck shuffles, restoring the process RNG afterward."""
    state_before = random.getstate()
    try:
        random.seed(int(seed))
        red_deck = PokemonTCGEnv._create_default_deck()
        blue_deck = PokemonTCGEnv._create_default_deck()
        return GameEngine.initialize_game("Red", red_deck, "Blue", blue_deck)
    finally:
        random.setstate(state_before)


def _action_mask(state: GameState, player_idx: int) -> np.ndarray:
    if player_idx not in (0, 1) or state.current_player_idx != player_idx:
        raise TournamentExecutionError("Requested a legal mask for a player who is not acting.")
    legal_actions = GameEngine.get_legal_actions(state)
    if not legal_actions:
        raise TournamentExecutionError("A live game state has no legal actions.")
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    for action in legal_actions:
        index = PokemonTCGEnv._action_to_index(action)
        if index is None or not 0 <= index < ACTION_DIM:
            raise TournamentExecutionError(f"Engine action has no valid 70-action encoding: {action!r}")
        mask[index] = 1.0
    if mask.sum() == 0:
        raise TournamentExecutionError("Engine legal actions produced an empty 70-action mask.")
    return mask


def _terminal_outcome(state: GameState) -> tuple[str | None, int | None]:
    """Classify terminal state using existing GameState information only."""
    if not state.game_over or state.winner_idx not in (0, 1):
        return None, None
    winner = int(state.winner_idx)
    loser = 1 - winner
    if state.knockout_counts[winner] >= 2:
        return "two_knockout_victory", None
    defeated = state.players[loser]
    if defeated.active is None and not defeated.has_bench():
        return "bench_out", loser
    if state.current_player_idx == loser and defeated.deck.remaining_count() == 0:
        return "deck_out", loser
    return "unclassified_terminal", None


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _game_rngs(config: TournamentConfig, plan: OrientationPlan, game_seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    common = (config.seed, game_seed, plan.match_type, plan.matchup, plan.orientation)
    red_seed = _stable_seed(*common, "red", plan.algorithm_red, plan.red_instance)
    blue_seed = _stable_seed(*common, "blue", plan.algorithm_blue, plan.blue_instance)
    return np.random.default_rng(red_seed), np.random.default_rng(blue_seed)


def run_single_game(
    plan: OrientationPlan,
    seed: int,
    max_decisions: int,
    red_agent: AgentAdapter,
    blue_agent: AgentAdapter,
    red_rng: np.random.Generator,
    blue_rng: np.random.Generator,
) -> dict[str, Any]:
    """Run one engine-level game; invalid adapter output aborts immediately."""
    state = _seeded_initial_state(seed)
    red_agent.reset()
    blue_agent.reset()
    agents = (red_agent, blue_agent)
    rngs = (red_rng, blue_rng)
    decisions = 0
    illegal_action_attempts = 0

    while not state.game_over and decisions < max_decisions:
        player_idx = int(state.current_player_idx)
        observation = get_phase2_information_set_observation(state, player_idx)
        if observation.shape != (OBSERVATION_DIM,):
            raise TournamentExecutionError(f"Player {player_idx} observation has invalid shape {observation.shape}.")
        action_mask = _action_mask(state, player_idx)
        # Only the acting player's view/mask and private agent-local recurrent
        # state reach the adapter. GameState and the other agent are not passed.
        action_index = agents[player_idx].select_action(observation, action_mask, rngs[player_idx])
        if not isinstance(action_index, (int, np.integer)) or not 0 <= int(action_index) < ACTION_DIM:
            illegal_action_attempts += 1
            raise IllegalActionError(
                f"{agents[player_idx].algorithm} returned invalid action {action_index!r} "
                f"for player index {player_idx} at decision {decisions}."
            )
        action_index = int(action_index)
        if action_mask[action_index] != 1.0:
            illegal_action_attempts += 1
            raise IllegalActionError(
                f"{agents[player_idx].algorithm} selected masked action {action_index} "
                f"for player index {player_idx} at decision {decisions}; no fallback was applied."
            )
        action = PokemonTCGEnv._index_to_action(action_index)
        try:
            state = GameEngine.step(state, action)
        except Exception as exc:
            raise TournamentExecutionError(
                f"The engine rejected legal-mask action {action_index} at decision {decisions}: {exc}"
            ) from exc
        decisions += 1

    terminal = bool(state.game_over)
    winner_idx = int(state.winner_idx) if terminal and state.winner_idx in (0, 1) else None
    terminal_reason, reason_player_idx = _terminal_outcome(state) if terminal else (None, None)
    red_reward = 1.0 if winner_idx == 0 else -1.0 if winner_idx == 1 else 0.0
    blue_reward = -red_reward
    return {
        "match_type": plan.match_type,
        "matchup": plan.matchup,
        "orientation": plan.orientation,
        "algorithm_red": plan.algorithm_red,
        "algorithm_blue": plan.algorithm_blue,
        "red_instance": plan.red_instance,
        "blue_instance": plan.blue_instance,
        "seed": int(seed),
        "decisions": decisions,
        "terminal": terminal,
        "step_cap_reached": not terminal and decisions >= max_decisions,
        "terminal_reason": terminal_reason or "step_cap",
        "winner": "Red" if winner_idx == 0 else "Blue" if winner_idx == 1 else "",
        "reason_player": "Red" if reason_player_idx == 0 else "Blue" if reason_player_idx == 1 else "",
        "red_reward": red_reward,
        "blue_reward": blue_reward,
        "red_knockouts": int(state.knockout_counts[0]),
        "blue_knockouts": int(state.knockout_counts[1]),
        "illegal_action_attempts": illegal_action_attempts,
    }


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["match_type"]),
        str(record["matchup"]),
        str(record["orientation"]),
        int(record["seed"]),
    )


def _read_game_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(record)
    for key in ("red_instance", "blue_instance", "seed", "decisions", "red_knockouts", "blue_knockouts", "illegal_action_attempts"):
        result[key] = int(result[key])
    for key in ("terminal", "step_cap_reached"):
        value = result[key]
        result[key] = value if isinstance(value, bool) else str(value).lower() == "true"
    for key in ("red_reward", "blue_reward"):
        result[key] = float(result[key])
    return result


def _append_game_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=GAME_RESULT_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        destination.flush()
        os.fsync(destination.fileno())


def _orientation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty orientation.")
    first = records[0]
    games = len(records)
    red_wins = sum(record["winner"] == "Red" for record in records)
    blue_wins = sum(record["winner"] == "Blue" for record in records)
    terminal_games = red_wins + blue_wins
    red_ci = wilson_interval(red_wins, terminal_games)
    blue_ci = wilson_interval(blue_wins, terminal_games)
    reason_count = lambda reason: sum(record["terminal_reason"] == reason for record in records)
    summary = {
        "match_type": first["match_type"],
        "matchup": first["matchup"],
        "orientation": first["orientation"],
        "algorithm_red": first["algorithm_red"],
        "algorithm_blue": first["algorithm_blue"],
        "games": games,
        "terminal_games": terminal_games,
        "red_wins": red_wins,
        "red_losses": blue_wins,
        "red_win_rate_percent": 100.0 * red_wins / terminal_games if terminal_games else 0.0,
        "red_win_rate_ci95_low_percent": red_ci[0],
        "red_win_rate_ci95_high_percent": red_ci[1],
        "blue_wins": blue_wins,
        "blue_losses": red_wins,
        "blue_win_rate_percent": 100.0 * blue_wins / terminal_games if terminal_games else 0.0,
        "blue_win_rate_ci95_low_percent": blue_ci[0],
        "blue_win_rate_ci95_high_percent": blue_ci[1],
        "red_reward_average": sum(float(record["red_reward"]) for record in records) / games,
        "blue_reward_average": sum(float(record["blue_reward"]) for record in records) / games,
        "average_decisions_per_game": sum(int(record["decisions"]) for record in records) / games,
        "red_two_knockout_victories": sum(record["terminal_reason"] == "two_knockout_victory" and record["winner"] == "Red" for record in records),
        "blue_two_knockout_victories": sum(record["terminal_reason"] == "two_knockout_victory" and record["winner"] == "Blue" for record in records),
        "red_deck_out_losses": sum(record["terminal_reason"] == "deck_out" and record["reason_player"] == "Red" for record in records),
        "blue_deck_out_losses": sum(record["terminal_reason"] == "deck_out" and record["reason_player"] == "Blue" for record in records),
        "red_bench_out_losses": sum(record["terminal_reason"] == "bench_out" and record["reason_player"] == "Red" for record in records),
        "blue_bench_out_losses": sum(record["terminal_reason"] == "bench_out" and record["reason_player"] == "Blue" for record in records),
        "nonterminal_at_step_cap": sum(bool(record["step_cap_reached"]) for record in records),
        "unclassified_terminal_games": reason_count("unclassified_terminal"),
        "illegal_action_attempts": sum(int(record["illegal_action_attempts"]) for record in records),
        "terminal_reason_counts": {
            reason: sum(record["terminal_reason"] == reason for record in records)
            for reason in ("two_knockout_victory", "bench_out", "deck_out", "unclassified_terminal", "step_cap")
        },
    }
    return summary


def _wilson_fields(wins: int, terminal_games: int) -> tuple[float, float, float]:
    low, high = wilson_interval(wins, terminal_games)
    rate = 100.0 * wins / terminal_games if terminal_games else 0.0
    return rate, low, high


def _cross_play_ranking(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cross_records = [record for record in records if record["match_type"] == "cross_play"]
    ratings = {algorithm: 1500.0 for algorithm in ALGORITHM_NAMES}
    k_factor = 32.0
    # Fixed plan/seed order makes the ordinary Elo update reproducible.
    for record in cross_records:
        red = record["algorithm_red"]
        blue = record["algorithm_blue"]
        red_score = 1.0 if record["winner"] == "Red" else 0.0 if record["winner"] == "Blue" else 0.5
        expected_red = 1.0 / (1.0 + 10.0 ** ((ratings[blue] - ratings[red]) / 400.0))
        delta = k_factor * (red_score - expected_red)
        ratings[red] += delta
        ratings[blue] -= delta

    ranking: list[dict[str, Any]] = []
    for algorithm in ALGORITHM_NAMES:
        games = [
            record for record in cross_records
            if record["algorithm_red"] == algorithm or record["algorithm_blue"] == algorithm
        ]
        wins = sum(
            (record["algorithm_red"] == algorithm and record["winner"] == "Red")
            or (record["algorithm_blue"] == algorithm and record["winner"] == "Blue")
            for record in games
        )
        losses = sum(
            (record["algorithm_red"] == algorithm and record["winner"] == "Blue")
            or (record["algorithm_blue"] == algorithm and record["winner"] == "Red")
            for record in games
        )
        nonterminal = sum(bool(record["step_cap_reached"]) for record in games)
        terminal = wins + losses
        low, high = wilson_interval(wins, terminal)
        average_reward = sum(
            float(record["red_reward"] if record["algorithm_red"] == algorithm else record["blue_reward"])
            for record in games
        ) / len(games) if games else 0.0
        ranking.append({
            "algorithm": algorithm,
            "games": len(games),
            "wins": wins,
            "losses": losses,
            "nonterminal_games": nonterminal,
            "completed_games": terminal,
            "win_rate_percent_completed": 100.0 * wins / terminal if terminal else 0.0,
            "win_rate_ci95_low_percent": low,
            "win_rate_ci95_high_percent": high,
            "score_percent_all_games_nonterminal_half": 100.0 * (wins + 0.5 * nonterminal) / len(games) if games else 0.0,
            "average_reward": average_reward,
            "elo": ratings[algorithm],
        })
    ranking.sort(key=lambda row: (-float(row["elo"]), str(row["algorithm"])))
    return ranking, ratings


def _pairwise_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    matrix: dict[str, dict[str, float | None]] = {}
    for row_algorithm in ALGORITHM_NAMES:
        row: dict[str, float | None] = {}
        for column_algorithm in ALGORITHM_NAMES:
            if row_algorithm == column_algorithm:
                row[column_algorithm] = None
                continue
            paired = [
                record for record in records
                if record["match_type"] == "cross_play"
                and {record["algorithm_red"], record["algorithm_blue"]} == {row_algorithm, column_algorithm}
            ]
            row_games = sum(record["terminal"] for record in paired)
            row_wins = sum(
                record["terminal"] and (
                    (record["algorithm_red"] == row_algorithm and record["winner"] == "Red")
                    or (record["algorithm_blue"] == row_algorithm and record["winner"] == "Blue")
                )
                for record in paired
            )
            row[column_algorithm] = 100.0 * row_wins / row_games if row_games else None
        matrix[row_algorithm] = row
    return matrix


def _flatten_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: row.get(key) for key in SUMMARY_FIELDS}
        for row in summary["orientation_results"]
    ]


def _write_reports(output_dir: Path, config_payload: dict[str, Any], records: list[dict[str, Any]], runtime_seconds: float) -> dict[str, Any]:
    records = [_normalise_record(record) for record in records]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["match_type"], record["matchup"], record["orientation"]), []).append(record)
    orientation_summaries = [
        _orientation_summary(grouped[key])
        for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2]))
    ]
    ranking, ratings = _cross_play_ranking(records)
    pairwise = _pairwise_matrix(records)
    self_play = [row for row in orientation_summaries if row["match_type"] == "self_play"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "config": config_payload,
        "games_completed": len(records),
        "expected_games": int(config_payload["expected_games"]),
        "complete": len(records) == int(config_payload["expected_games"]),
        "runtime_seconds_this_invocation": float(runtime_seconds),
        "orientation_results": orientation_summaries,
        "self_play_results": self_play,
        "pairwise_win_rate_matrix_percent_completed_games": pairwise,
        "cross_play_ranking": ranking,
        "elo_k_factor": 32.0,
        "elo_ratings": ratings,
        "nonterminal_policy": "Capped games remain nonterminal (no win/loss); they are reported separately and score 0.5 only for Elo/all-game score.",
        "confidence_interval_policy": "95% Wilson intervals use completed terminal games as the denominator; capped nonterminal games are reported separately.",
    }
    _atomic_write(output_dir / "tournament_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write_csv(output_dir / "tournament_summary.csv", SUMMARY_FIELDS, _flatten_summary(summary))

    matrix_path = output_dir / "pairwise_win_matrix.csv"
    matrix_fields = ("algorithm",) + ALGORITHM_NAMES
    _atomic_write_csv(matrix_path, matrix_fields, [
        {"algorithm": algorithm, **pairwise[algorithm]} for algorithm in ALGORITHM_NAMES
    ])
    _atomic_write_csv(
        output_dir / "self_play_results.csv",
        SUMMARY_FIELDS,
        [{key: row.get(key) for key in SUMMARY_FIELDS} for row in self_play],
    )

    report_lines = [
        "FOUR-ALGORITHM HEAD-TO-HEAD AND SELF-PLAY TOURNAMENT",
        "=" * 58,
        f"Run directory: {output_dir}",
        f"Games completed: {len(records)} / {config_payload['expected_games']}",
        f"Runtime this invocation: {runtime_seconds:.2f} seconds",
        f"Decision cap: {config_payload['max_decisions']} total decisions by both players",
        "Win-rate CIs use terminal games only; step-cap games remain nonterminal.",
        "",
        "ROLE-SEPARATED MATCHUP RESULTS",
        "-" * 58,
        "Matchup | Orientation | Games | Red W | Blue W | Red win% [95% CI] | Nonterminal | Avg decisions",
    ]
    for row in orientation_summaries:
        report_lines.append(
            f"{row['matchup']} | {row['orientation']} | {row['games']} | {row['red_wins']} | "
            f"{row['blue_wins']} | {row['red_win_rate_percent']:.1f}% "
            f"[{row['red_win_rate_ci95_low_percent']:.1f}, {row['red_win_rate_ci95_high_percent']:.1f}] | "
            f"{row['nonterminal_at_step_cap']} | {row['average_decisions_per_game']:.1f}"
        )
    report_lines.extend([
        "",
        "PAIRWISE WIN-RATE MATRIX (% of completed games; capped games excluded)",
        "-" * 58,
        "Algorithm | " + " | ".join(ALGORITHM_NAMES),
    ])
    for algorithm in ALGORITHM_NAMES:
        values = ["—" if pairwise[algorithm][opponent] is None else f"{pairwise[algorithm][opponent]:.1f}" for opponent in ALGORITHM_NAMES]
        report_lines.append(algorithm + " | " + " | ".join(values))
    report_lines.extend([
        "",
        "SELF-PLAY ROLE RESULTS",
        "-" * 58,
    ])
    for row in self_play:
        report_lines.append(
            f"{row['matchup']} | {row['orientation']} | Red wins {row['red_wins']}/{row['terminal_games']} | "
            f"Blue wins {row['blue_wins']}/{row['terminal_games']} | capped {row['nonterminal_at_step_cap']} | "
            f"avg decisions {row['average_decisions_per_game']:.1f}"
        )
    report_lines.extend([
        "",
        "OVERALL CROSS-PLAY RANKING (descriptive; not compute-matched)",
        "-" * 58,
        "Rank | Algorithm | Wins | Losses | Capped | Completed win% | 95% CI | Score% incl. capped half | Elo",
    ])
    for rank, row in enumerate(ranking, start=1):
        report_lines.append(
            f"{rank} | {row['algorithm']} | {row['wins']} | {row['losses']} | {row['nonterminal_games']} | "
            f"{row['win_rate_percent_completed']:.1f}% | [{row['win_rate_ci95_low_percent']:.1f}, "
            f"{row['win_rate_ci95_high_percent']:.1f}] | {row['score_percent_all_games_nonterminal_half']:.1f}% | "
            f"{row['elo']:.1f}"
        )
    report_lines.extend([
        "",
        "Elo uses K=32, starts at 1500, processes cross-play game records in fixed order, and scores capped games as draws.",
        "Self-play results describe competition against the same policy family; they do not establish equilibrium or exploitability.",
    ])
    _atomic_write(output_dir / "tournament_report.txt", "\n".join(report_lines) + "\n")
    return summary


def _checkpoint_metadata() -> dict[str, Any]:
    checkpoint_paths: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for algorithm, roles in DEFAULT_CHECKPOINTS.items():
        checkpoint_paths[algorithm] = dict(roles)
        for role, relative in roles.items():
            absolute = PROJECT_ROOT / relative
            if not absolute.is_file():
                raise FileNotFoundError(f"Required {algorithm} checkpoint ({role}) not found: {absolute}")
            digest = hashlib.sha256()
            with absolute.open("rb") as checkpoint:
                for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[f"{algorithm}:{role}"] = digest.hexdigest()
    return {"relative_paths": checkpoint_paths, "sha256": hashes}


def _configuration_payload(config: TournamentConfig) -> dict[str, Any]:
    config.validate()
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **asdict(config),
        "expected_games": config.expected_games,
        "observation_dimension": OBSERVATION_DIM,
        "action_dimension": ACTION_DIM,
        "game_step_definition": "one GameEngine.step by either player; maximum counts Red and Blue decisions together",
        "seed_policy": {
            "game_deck_seed": "base seed + game index; paired across orientations and matchups",
            "policy_rng": "stable BLAKE2-derived independent NumPy Generator for each game, orientation, seat, algorithm, and player instance",
            "python_random": "temporarily seeded only for initialization deck shuffles and restored immediately afterward",
        },
        "policy_modes": {
            "DQN": "greedy masked Q-values, epsilon=0",
            "PPO": "masked categorical probabilities, deterministic argmax",
            "LSTM-PPO": "masked categorical probabilities, deterministic argmax, independent per-seat hidden/cell state",
            "Deep CFR": "sample from current legal, normalized StrategyNetwork average-strategy probabilities",
        },
        "deep_cfr_seat_assignment": "Red StrategyNetwork plays Red; Blue StrategyNetwork plays Blue in every orientation and self-play condition",
        "checkpoints": _checkpoint_metadata(),
    }


def _load_saved_config(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "tournament_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume: config file is missing at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_resume_compatible(saved: dict[str, Any], config: TournamentConfig) -> dict[str, Any]:
    for key, value in asdict(config).items():
        if saved.get(key) != value:
            raise ValueError(
                f"Resume configuration mismatch for {key}: saved {saved.get(key)!r}, requested {value!r}."
            )
    if saved.get("expected_games") != config.expected_games:
        raise ValueError("Resume expected-game count does not match the saved configuration.")
    current_checkpoints = _checkpoint_metadata()
    if saved.get("checkpoints") != current_checkpoints:
        raise ValueError("Checkpoint paths or contents changed since this tournament run began.")
    return saved


def _orientation_agents(
    plan: OrientationPlan,
    device: torch.device,
    agent_factory: Callable[[str, int, torch.device], AgentAdapter],
) -> tuple[AgentAdapter, AgentAdapter]:
    red_agent = agent_factory(plan.algorithm_red, 0, device)
    blue_agent = agent_factory(plan.algorithm_blue, 1, device)
    if red_agent is blue_agent:
        raise TournamentExecutionError("Red and Blue must have independent adapter instances.")
    return red_agent, blue_agent


def run_tournament(
    config: TournamentConfig,
    *,
    output_directory: str | Path | None = None,
    resume: bool = False,
    agent_factory: Callable[[str, int, torch.device], AgentAdapter] = load_agent,
) -> tuple[Path, dict[str, Any]]:
    """Run selected tournament orientations and atomically publish reports.

    A resume reads completed game keys from the existing CSV and skips them.
    The saved game seeds and checkpoint hashes must still match.
    """
    config.validate()
    started_at = time.perf_counter()
    if resume:
        if output_directory is None:
            raise ValueError("Resume requires the existing output_directory.")
        run_dir = Path(output_directory).expanduser().resolve()
        saved_payload = _assert_resume_compatible(_load_saved_config(run_dir), config)
        config_payload = saved_payload
    else:
        run_dir = (
            Path(output_directory).expanduser().resolve()
            if output_directory is not None
            else create_unique_run_directory(seed=config.seed)
        )
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite a non-empty tournament directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config_payload = _configuration_payload(config)
        _atomic_write(run_dir / "tournament_config.json", json.dumps(config_payload, indent=2, sort_keys=True) + "\n")

    game_results_path = run_dir / GAME_RESULTS_FILENAME
    records = [_normalise_record(record) for record in _read_game_records(game_results_path)]
    completed_keys = {_record_key(record) for record in records}
    if len(completed_keys) != len(records):
        raise ValueError("Tournament game-level results contain duplicate completed-game keys.")
    expected_keys: set[tuple[str, str, str, int]] = set()
    for plan in build_orientation_plans(config.mode):
        pending_seeds: list[int] = []
        for game_index in range(config.games_per_orientation):
            seed = config.seed + game_index
            key = (plan.match_type, plan.matchup, plan.orientation, seed)
            expected_keys.add(key)
            if key in completed_keys:
                continue
            pending_seeds.append(seed)
        if not pending_seeds:
            continue
        red_agent, blue_agent = _orientation_agents(plan, torch.device(config.device), agent_factory)
        # Reuse two independent seat instances for each game's same
        # orientation. reset() isolates recurrent state at each game start.
        for seed in pending_seeds:
            red_rng, blue_rng = _game_rngs(config, plan, seed)
            record = run_single_game(
                plan,
                seed,
                config.max_decisions,
                red_agent,
                blue_agent,
                red_rng,
                blue_rng,
            )
            _append_game_record(game_results_path, record)
            records.append(record)
            completed_keys.add((plan.match_type, plan.matchup, plan.orientation, seed))
    extra_keys = completed_keys - expected_keys
    if extra_keys:
        raise ValueError(f"Run directory contains game records outside the saved schedule: {sorted(extra_keys)[:3]}")
    if len(completed_keys) != len(expected_keys):
        raise TournamentExecutionError(
            f"Only {len(completed_keys)} of {len(expected_keys)} scheduled games are recorded."
        )
    elapsed = time.perf_counter() - started_at
    summary = _write_reports(run_dir, config_payload, records, elapsed)
    return run_dir, summary

