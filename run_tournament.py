"""Run the four-algorithm head-to-head and self-play tournament."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pokemon_rl.evaluation.tournament import (  # noqa: E402
    RESULTS_ROOT,
    TournamentConfig,
    run_tournament,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN, PPO, LSTM-PPO, and Deep CFR head-to-head and in self-play.")
    parser.add_argument("--games-per-orientation", type=int, default=None, help="Games for each matchup/seat assignment (default: 200).")
    parser.add_argument("--max-decisions", type=int, default=None, help="Maximum total Red+Blue engine decisions per game (default: 200).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="Inference device (default: auto; CUDA when available).")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for paired game seeds and independent policy RNGs (default: 12345).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cross-play-only", action="store_true", help="Run the six cross-algorithm matchups only.")
    mode.add_argument("--self-play-only", action="store_true", help="Run the four self-play matchups only.")
    parser.add_argument(
        "--resume", nargs="?", const="latest", metavar="RUN_DIR",
        help="Resume an existing run directory, or the latest incomplete run when no path is given.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Set a new output directory; never overwrites a non-empty directory.")
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    config_path = path / "tournament_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No tournament_config.json found in {path}.")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _latest_incomplete_run() -> Path:
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"No tournament result directories found under {RESULTS_ROOT}.")
    candidates = sorted(
        (path for path in RESULTS_ROOT.iterdir() if path.is_dir() and (path / "tournament_config.json").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        saved = _load_config(path)
        expected = int(saved.get("expected_games", -1))
        games_path = path / "game_level_results.csv"
        if not games_path.is_file():
            return path
        with games_path.open("r", encoding="utf-8", newline="") as source:
            completed = max(0, sum(1 for _ in csv.DictReader(source)))
        if expected >= 0 and completed < expected:
            return path
    raise FileNotFoundError("No incomplete tournament run was found to resume.")


def _resume_config(saved: dict, args: argparse.Namespace) -> TournamentConfig:
    selected_mode = "cross_play_only" if args.cross_play_only else "self_play_only" if args.self_play_only else saved["mode"]
    requested = {
        "games_per_orientation": args.games_per_orientation,
        "max_decisions": args.max_decisions,
        "seed": args.seed,
        "mode": selected_mode,
    }
    for key, value in requested.items():
        if value is not None and value != saved[key]:
            raise ValueError(f"Cannot resume with a changed {key}: saved={saved[key]!r}, requested={value!r}.")
    if args.device is not None and args.device != "auto" and args.device != saved["device"]:
        raise ValueError(f"Cannot resume with a changed device: saved={saved['device']!r}, requested={args.device!r}.")
    return TournamentConfig(
        games_per_orientation=int(saved["games_per_orientation"]),
        max_decisions=int(saved["max_decisions"]),
        seed=int(saved["seed"]),
        device=str(saved["device"]),
        mode=str(saved["mode"]),
    )


def _new_config(args: argparse.Namespace) -> TournamentConfig:
    device = args.device or "auto"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports that CUDA is unavailable.")
    return TournamentConfig(
        games_per_orientation=200 if args.games_per_orientation is None else args.games_per_orientation,
        max_decisions=200 if args.max_decisions is None else args.max_decisions,
        seed=12345 if args.seed is None else args.seed,
        device=device,
        mode="cross_play_only" if args.cross_play_only else "self_play_only" if args.self_play_only else "all",
    )


def main() -> None:
    args = _parse_args()
    if args.resume is not None:
        if args.output_dir is not None:
            raise ValueError("Use --resume RUN_DIR instead of --output-dir when resuming.")
        run_dir = _latest_incomplete_run() if args.resume == "latest" else Path(args.resume).expanduser().resolve()
        config = _resume_config(_load_config(run_dir), args)
        print(f"Resuming tournament in {run_dir}")
        run_dir, summary = run_tournament(config, output_directory=run_dir, resume=True)
    else:
        config = _new_config(args)
        print(f"Starting {config.mode} tournament: {config.expected_games} games on {config.device}.")
        run_dir, summary = run_tournament(config, output_directory=args.output_dir, resume=False)

    print(f"Completed {summary['games_completed']} / {summary['expected_games']} games.")
    print(f"Tournament outputs: {run_dir}")
    print(f"Report: {run_dir / 'tournament_report.txt'}")
    print(f"Runtime this invocation: {summary['runtime_seconds_this_invocation']:.2f} seconds")


if __name__ == "__main__":
    main()

