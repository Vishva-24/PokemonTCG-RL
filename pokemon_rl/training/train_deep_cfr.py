"""True two-player Deep CFR training for the Mini Pokemon TCG.

This module deliberately uses :class:`GameEngine` directly. ``PokemonTCGEnv``
is a single-agent wrapper that advances a built-in random Blue player, so it is
appropriate for the post-training benchmark but not for CFR traversal.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim

from pokemon_rl.agents.deep_cfr import (
    RegretNetwork,
    ReservoirBuffer,
    StrategyNetwork,
    StrategyCoverageAudit,
    train_regret_network,
    train_strategy_network,
)
from pokemon_rl.agents.deep_cfr_traversal import run_cfr_traversal
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.pokemon import Attack, PokemonCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM
from pokemon_rl.game import GameEngine
from pokemon_rl.player import Deck


RED = 0
BLUE = 1
PLAYER_NAMES = {RED: "red", BLUE: "blue"}


@dataclass
class DeepCFRConfig:
    """Hyperparameters for a full Deep CFR run.

    Defaults are the requested real-experiment configuration. Use
    :meth:`rapid` for the deliberately small validation configuration.
    """

    num_iterations: int = 100
    traversals_per_iter: int = 128
    batch_size: int = 128
    reservoir_capacity: int = 40_000
    learning_rate: float = 1e-4
    regret_epochs: int = 32
    strategy_epochs: int = 32
    # Expanded-tree horizon. At this limit the traversal samples a real terminal
    # continuation from the current regret-matching profile; it never invents a
    # truncated-leaf utility. Four was selected from the real-deck profiler.
    max_steps: int = 4
    checkpoint_dir: str = "results"
    seed: int = 4_000
    device: Optional[Union[str, torch.device]] = None
    compact_test_deck: bool = False
    # Optional passive metadata collection. It observes strategy sample
    # insertion after normal reservoir replacement and is off by default.
    strategy_coverage_audit: bool = False

    @classmethod
    def rapid(cls, checkpoint_dir: str = "results", seed: int = 4_000) -> "DeepCFRConfig":
        """Return the requested two-iteration test configuration."""
        return cls(
            num_iterations=2,
            traversals_per_iter=4,
            batch_size=16,
            reservoir_capacity=1_000,
            regret_epochs=2,
            strategy_epochs=2,
            # Retain the production bounded/approximate MCCFR horizon. The
            # compact deck keeps rapid tests small without changing Phase 2
            # game rules or replacing the real continuation rollout.
            max_steps=4,
            # The six-card deck reaches terminal states quickly while keeping
            # Phase 2 Bench, typed-Energy, KO, and deck-out state semantics.
            compact_test_deck=True,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
        )

    @classmethod
    def real(cls, checkpoint_dir: str = "results", seed: int = 4_000) -> "DeepCFRConfig":
        """Return the requested 100-iteration experiment configuration."""
        return cls(checkpoint_dir=checkpoint_dir, seed=seed)

    def validate(self) -> None:
        for field_name in (
            "num_iterations", "traversals_per_iter", "batch_size", "reservoir_capacity",
            "regret_epochs", "strategy_epochs", "max_steps",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        """Match the configuration-reporting interface of the other trainers."""
        return {
            "num_iterations": self.num_iterations,
            "traversals_per_iter": self.traversals_per_iter,
            "batch_size": self.batch_size,
            "reservoir_capacity": self.reservoir_capacity,
            "learning_rate": self.learning_rate,
            "regret_epochs": self.regret_epochs,
            "strategy_epochs": self.strategy_epochs,
            "max_steps": self.max_steps,
            "checkpoint_dir": self.checkpoint_dir,
            "seed": self.seed,
            "device": str(self.device) if self.device is not None else "auto",
            "compact_test_deck": self.compact_test_deck,
            "strategy_coverage_audit": self.strategy_coverage_audit,
        }


class DeepCFRTrainer:
    """Run external-sampling MCCFR for Red and Blue with separate memories.

    ``best`` strategy checkpoints use the lowest Red iteration-weighted masked
    strategy cross-entropy observed at an outer iteration. This is a fitting
    diagnostic, not a gameplay claim; the 1,500-game post-training benchmark
    determines playing strength.
    """

    BEST_CHECKPOINT_CRITERION = "lowest Red iteration-weighted masked strategy cross-entropy diagnostic"
    # The strategy-learning objective uses masked distribution cross-entropy
    # with outer-iteration sample weights. A distinct suffix prevents an older
    # unweighted strategy checkpoint from being silently evaluated as this run.
    STRATEGY_CHECKPOINT_KIND = "strategy_masked_ce_iteration_weighted"

    def __init__(self, config: DeepCFRConfig):
        config.validate()
        self.config = config
        self.device = self._resolve_device(config.device)
        self._seed_everything(config.seed)

        # Separate regret and average-strategy approximators for the two sides.
        self.regret_networks = {
            RED: self._move_to_device(RegretNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)),
            BLUE: self._move_to_device(RegretNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)),
        }
        self.strategy_networks = {
            RED: self._move_to_device(StrategyNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)),
            BLUE: self._move_to_device(StrategyNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)),
        }
        self.regret_optimizers = {
            player: optim.Adam(network.parameters(), lr=config.learning_rate)
            for player, network in self.regret_networks.items()
        }
        self.strategy_optimizers = {
            player: optim.Adam(network.parameters(), lr=config.learning_rate)
            for player, network in self.strategy_networks.items()
        }

        # Four distinct reservoirs; no Red/Blue or regret/strategy sample sharing.
        self.regret_buffers = {
            RED: ReservoirBuffer(config.reservoir_capacity, seed=config.seed + 101,
                                 state_dim=OBSERVATION_DIM, target_dim=ACTION_DIM),
            BLUE: ReservoirBuffer(config.reservoir_capacity, seed=config.seed + 102,
                                  state_dim=OBSERVATION_DIM, target_dim=ACTION_DIM),
        }
        self.strategy_buffers = {
            RED: ReservoirBuffer(config.reservoir_capacity, seed=config.seed + 201,
                                 state_dim=OBSERVATION_DIM, target_dim=ACTION_DIM,
                                 mask_dim=ACTION_DIM, sample_weight=True),
            BLUE: ReservoirBuffer(config.reservoir_capacity, seed=config.seed + 202,
                                  state_dim=OBSERVATION_DIM, target_dim=ACTION_DIM,
                                  mask_dim=ACTION_DIM, sample_weight=True),
        }
        self.strategy_coverage_audit = (
            StrategyCoverageAudit(config.reservoir_capacity)
            if config.strategy_coverage_audit
            else None
        )
        self.strategy_coverage_audit_paths: Optional[Dict[str, str]] = None
        self.traversal_records: List[Dict[str, Any]] = []
        self.best_checkpoint_score = math.inf

    @staticmethod
    def _resolve_device(requested: Optional[Union[str, torch.device]]) -> torch.device:
        if requested is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return device

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed every source used by root setup, traversal sampling, and SGD."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _move_to_device(self, network: Union[RegretNetwork, StrategyNetwork]):
        """Keep the legacy network ``device`` attribute in sync with ``.to``."""
        network.to(self.device)
        network.device = self.device
        return network

    def print_hardware_info(self) -> None:
        """Print the requested device report without assuming a GPU exists."""
        print("=" * 70)
        print("                    DEEP CFR HARDWARE DETECTION")
        print("=" * 70)
        print(f"PyTorch version:   {torch.__version__}")
        print(f"CUDA available:    {torch.cuda.is_available()}")
        print(f"CUDA version:      {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"GPU name:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (CPU only)'}")
        print(f"Selected device:   {self.device}")
        print("=" * 70)

    @staticmethod
    def _create_default_deck() -> Deck:
        """Build the exact 20-card Phase 2 deck for an isolated CFR root."""
        return Deck([
            PokemonCard(
                "Pikachu", 70, card_id=948, expansion="ASC", collection_no=55,
                pokemon_type="Lightning", weakness="Fighting", retreat_cost=1,
                attacks=[
                    Attack("Tail Smack", {"Lightning": 1}, 10, "Lightning"),
                    Attack("Tiny Bolt", {"Lightning": 1, "Colorless": 1}, 30, "Lightning"),
                ],
            ),
            PokemonCard(
                "Charmander", 80, card_id=926, expansion="ASC", collection_no=20,
                pokemon_type="Fire", weakness="Water", retreat_cost=1,
                attacks=[Attack("Fire Claws", {"Fire": 1, "Colorless": 1}, 30, "Fire")],
            ),
            PokemonCard(
                "Swinub", 70, card_id=800, expansion="PFL", collection_no=23,
                pokemon_type="Water", weakness="Metal", retreat_cost=2,
                attacks=[
                    Attack("Stampede", {"Colorless": 1}, 10, "Water"),
                    Attack("Icy Snow", {"Water": 1, "Colorless": 1}, 20, "Water"),
                ],
            ),
            PokemonCard(
                "Applin", 40, card_id=149, expansion="SCR", collection_no=12,
                pokemon_type="Grass", weakness="Fire", retreat_cost=1,
                attacks=[Attack("Spray Fluid", {"Grass": 1}, 20, "Grass")],
            ),
            *[EnergyCard("Basic Grass Energy", "Grass") for _ in range(3)],
            *[EnergyCard("Basic Fire Energy", "Fire") for _ in range(3)],
            *[EnergyCard("Basic Water Energy", "Water") for _ in range(3)],
            *[EnergyCard("Basic Lightning Energy", "Lightning") for _ in range(3)],
            *[TrainerCard("Potion", "heal", 30, card_id=1117) for _ in range(4)],
        ])

    @staticmethod
    def _create_compact_test_deck() -> Deck:
        """Create the small, terminal-reaching deck reserved for rapid tests.

        This changes neither the game engine nor any Phase 2 rule. It only
        keeps the rapid validation tree finite; real training always uses the
        exact 20-card deck from :meth:`_create_default_deck`.
        """
        return Deck([
            PokemonCard(
                "Charmander", 80, card_id=926, expansion="ASC", collection_no=20,
                pokemon_type="Fire", weakness="Water", retreat_cost=1,
                attacks=[Attack("Fire Claws", {"Fire": 1, "Colorless": 1}, 30, "Fire")],
            ),
            PokemonCard(
                "Applin", 40, card_id=149, expansion="SCR", collection_no=12,
                pokemon_type="Grass", weakness="Fire", retreat_cost=1,
                attacks=[Attack("Spray Fluid", {"Grass": 1}, 20, "Grass")],
            ),
            *[EnergyCard("Basic Fire Energy", "Fire") for _ in range(3)],
            TrainerCard("Potion", "heal", 30, card_id=1117),
        ])

    def _traversal_seed(self, iteration: int, traverser: int, traversal_number: int) -> int:
        """Return a unique deterministic seed for one self-play traversal."""
        return self.config.seed + iteration * 100_000 + traverser * 10_000 + traversal_number

    def _fresh_root_state(self):
        deck_factory = self._create_compact_test_deck if self.config.compact_test_deck else self._create_default_deck
        return GameEngine.initialize_game(
            "Red", deck_factory(), "Blue", deck_factory()
        )

    def _run_player_traversals(self, iteration: int, traverser: int, verbose: bool = False) -> Dict[str, Any]:
        """Run one role's traversals from fresh directly-managed game roots."""
        utilities: List[float] = []
        decision_nodes = terminal_nodes = regret_samples = strategy_samples = 0
        for traversal_number in range(self.config.traversals_per_iter):
            seed = self._traversal_seed(iteration, traverser, traversal_number)
            self._seed_everything(seed)
            report_this_traversal = verbose and (
                traversal_number == 0
                or (traversal_number + 1) % max(1, self.config.traversals_per_iter // 8) == 0
                or traversal_number + 1 == self.config.traversals_per_iter
            )
            if report_this_traversal:
                print(
                    f"  {PLAYER_NAMES[traverser].title()} traversal "
                    f"{traversal_number + 1}/{self.config.traversals_per_iter} started "
                    f"(horizon={self.config.max_steps})...",
                    flush=True,
                )

            def report_nodes(snapshot: Dict[str, Any]) -> None:
                if report_this_traversal:
                    print(
                        f"    {PLAYER_NAMES[traverser].title()} traversal {traversal_number + 1}: "
                        f"{snapshot['decision_nodes']} decision nodes, "
                        f"depth {snapshot['max_depth']}, "
                        f"{snapshot['state_copies']} snapshots",
                        flush=True,
                    )

            utility, diagnostics = run_cfr_traversal(
                root_state=self._fresh_root_state(),
                traverser=traverser,
                regret_net=self.regret_networks,
                strategy_net=self.strategy_networks,
                # Scalar buffers prevent accidental cross-player writes.
                regret_buffer=self.regret_buffers[traverser],
                strategy_buffer=self.strategy_buffers[traverser],
                iteration=iteration,
                max_steps=self.config.max_steps,
                progress_every_nodes=250 if report_this_traversal else 0,
                progress_callback=report_nodes if report_this_traversal else None,
                strategy_coverage_audit=self.strategy_coverage_audit,
            )
            utility = float(utility)
            if not math.isfinite(utility) or not -1.0 <= utility <= 1.0:
                raise RuntimeError(f"Invalid traversal utility {utility} for player {traverser}.")
            utilities.append(utility)
            decision_nodes += int(diagnostics["decision_nodes"])
            terminal_nodes += int(diagnostics["terminal_nodes"])
            regret_samples += int(diagnostics["regret_samples"])
            strategy_samples += int(diagnostics["strategy_samples"])
            self.traversal_records.append({
                "iteration": iteration, "traverser": traverser,
                "traversal": traversal_number, "seed": seed,
                "utility": utility, "diagnostics": dict(diagnostics),
            })
            if report_this_traversal:
                print(
                    f"  {PLAYER_NAMES[traverser].title()} traversal {traversal_number + 1}/"
                    f"{self.config.traversals_per_iter} complete: "
                    f"{diagnostics['elapsed_seconds']:.2f}s, "
                    f"{diagnostics['decision_nodes']} decision nodes, "
                    f"{diagnostics['horizon_rollouts']} continuation rollouts.",
                    flush=True,
                )
        return {
            "utility": float(np.mean(utilities)), "decision_nodes": decision_nodes,
            "terminal_nodes": terminal_nodes, "regret_samples": regret_samples,
            "strategy_samples": strategy_samples,
        }

    def _train_player_networks(self, player: int) -> Dict[str, float]:
        """Fit a player's networks exclusively from that player's reservoirs."""
        regret_losses: List[float] = []
        strategy_losses: List[float] = []
        # ReservoirBuffer supports replacement sampling, so early non-empty
        # memories can still use the configured batch size.
        if self.regret_buffers[player]:
            for _ in range(self.config.regret_epochs):
                regret_losses.append(float(train_regret_network(
                    self.regret_networks[player], self.regret_optimizers[player],
                    self.regret_buffers[player].sample(self.config.batch_size),
                )))
        if self.strategy_buffers[player]:
            for _ in range(self.config.strategy_epochs):
                strategy_losses.append(float(train_strategy_network(
                    self.strategy_networks[player], self.strategy_optimizers[player],
                    self.strategy_buffers[player].sample(self.config.batch_size),
                )))
        regret_loss = float(np.mean(regret_losses)) if regret_losses else 0.0
        strategy_loss = float(np.mean(strategy_losses)) if strategy_losses else 0.0
        if not math.isfinite(regret_loss) or not math.isfinite(strategy_loss):
            raise RuntimeError(f"Non-finite network loss for player {player}.")
        return {"regret_loss": regret_loss, "strategy_loss": strategy_loss}

    def _checkpoint_path(self, stage: str, player: int, network_kind: str) -> str:
        if network_kind == "strategy":
            network_kind = self.STRATEGY_CHECKPOINT_KIND
        return os.path.join(
            self.config.checkpoint_dir,
            f"{stage}_deep_cfr_{PLAYER_NAMES[player]}_{network_kind}.pt",
        )

    def _save_network_set(self, stage: str) -> None:
        for player in (RED, BLUE):
            torch.save(self.strategy_networks[player].state_dict(), self._checkpoint_path(stage, player, "strategy"))
            torch.save(self.regret_networks[player].state_dict(), self._checkpoint_path(stage, player, "regret"))

    @staticmethod
    def _empty_history() -> Dict[str, List[Any]]:
        columns = [
            "iteration", "red_traversal_utility", "blue_traversal_utility",
            "red_decision_nodes", "blue_decision_nodes", "red_terminal_nodes", "blue_terminal_nodes",
            "red_regret_samples", "blue_regret_samples", "red_strategy_samples", "blue_strategy_samples",
            "red_regret_network_loss", "blue_regret_network_loss",
            "red_strategy_network_loss", "blue_strategy_network_loss",
            "red_regret_reservoir_size", "blue_regret_reservoir_size",
            "red_strategy_reservoir_size", "blue_strategy_reservoir_size",
            "elapsed_training_time", "best_checkpoint_score",
            # Short aliases retain compatibility with the earlier partial loop.
            "red_utility", "blue_utility", "red_regret_loss", "blue_regret_loss",
            "red_strategy_loss", "blue_strategy_loss", "red_regret_mem_size", "blue_regret_mem_size",
            "red_strategy_mem_size", "blue_strategy_mem_size", "decision_nodes", "terminal_nodes", "elapsed_time",
        ]
        return {column: [] for column in columns}

    @staticmethod
    def _append_metrics(history: Dict[str, List[Any]], metrics: Dict[str, Any]) -> None:
        for column in history:
            history[column].append(metrics[column])

    def _write_metrics(self, history: Dict[str, List[Any]]) -> str:
        path = os.path.join(self.config.checkpoint_dir, "deep_cfr_training_metrics.csv")
        columns = list(history)
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            for index in range(len(history["iteration"])):
                writer.writerow({column: history[column][index] for column in columns})
        return path

    def train(self, verbose: bool = True, show_hardware: bool = True) -> Dict[str, List[Any]]:
        """Train both CFR players and save metrics plus final checkpoints."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        if verbose and show_hardware:
            self.print_hardware_info()
        if verbose:
            print(f"Best checkpoint criterion: {self.BEST_CHECKPOINT_CRITERION}; it is not a gameplay metric.")
        history = self._empty_history()
        for iteration in range(1, self.config.num_iterations + 1):
            iteration_start = time.perf_counter()
            # Both traversals are CFR self-play through GameEngine, never PokemonTCGEnv.
            if verbose:
                print(f"Iteration {iteration}/{self.config.num_iterations}: running Red CFR traversals...", flush=True)
            red = self._run_player_traversals(iteration, RED, verbose=verbose)
            if verbose:
                print(f"Iteration {iteration}/{self.config.num_iterations}: running Blue CFR traversals...", flush=True)
            blue = self._run_player_traversals(iteration, BLUE, verbose=verbose)
            red_losses = self._train_player_networks(RED)
            blue_losses = self._train_player_networks(BLUE)

            checkpoint_score = red_losses["strategy_loss"]
            if checkpoint_score <= self.best_checkpoint_score:
                self.best_checkpoint_score = checkpoint_score
                self._save_network_set("best")

            elapsed = time.perf_counter() - iteration_start
            metrics = {
                "iteration": iteration,
                "red_traversal_utility": red["utility"], "blue_traversal_utility": blue["utility"],
                "red_decision_nodes": red["decision_nodes"], "blue_decision_nodes": blue["decision_nodes"],
                "red_terminal_nodes": red["terminal_nodes"], "blue_terminal_nodes": blue["terminal_nodes"],
                "red_regret_samples": red["regret_samples"], "blue_regret_samples": blue["regret_samples"],
                "red_strategy_samples": red["strategy_samples"], "blue_strategy_samples": blue["strategy_samples"],
                "red_regret_network_loss": red_losses["regret_loss"],
                "blue_regret_network_loss": blue_losses["regret_loss"],
                "red_strategy_network_loss": red_losses["strategy_loss"],
                "blue_strategy_network_loss": blue_losses["strategy_loss"],
                "red_regret_reservoir_size": len(self.regret_buffers[RED]),
                "blue_regret_reservoir_size": len(self.regret_buffers[BLUE]),
                "red_strategy_reservoir_size": len(self.strategy_buffers[RED]),
                "blue_strategy_reservoir_size": len(self.strategy_buffers[BLUE]),
                "elapsed_training_time": elapsed, "best_checkpoint_score": self.best_checkpoint_score,
                "red_utility": red["utility"], "blue_utility": blue["utility"],
                "red_regret_loss": red_losses["regret_loss"], "blue_regret_loss": blue_losses["regret_loss"],
                "red_strategy_loss": red_losses["strategy_loss"], "blue_strategy_loss": blue_losses["strategy_loss"],
                "red_regret_mem_size": len(self.regret_buffers[RED]),
                "blue_regret_mem_size": len(self.regret_buffers[BLUE]),
                "red_strategy_mem_size": len(self.strategy_buffers[RED]),
                "blue_strategy_mem_size": len(self.strategy_buffers[BLUE]),
                "decision_nodes": red["decision_nodes"] + blue["decision_nodes"],
                "terminal_nodes": red["terminal_nodes"] + blue["terminal_nodes"],
                "elapsed_time": elapsed,
            }
            self._append_metrics(history, metrics)
            if verbose:
                print(
                    f"Iteration {iteration:3d} | Red utility {red['utility']:+.3f} | "
                    f"Blue utility {blue['utility']:+.3f} | "
                    f"R/B regret loss {red_losses['regret_loss']:.6f}/{blue_losses['regret_loss']:.6f} | "
                    f"R/B strategy loss {red_losses['strategy_loss']:.6f}/{blue_losses['strategy_loss']:.6f} | "
                    f"time {elapsed:.2f}s"
                )

        self._save_network_set("final")
        if self.strategy_coverage_audit is not None:
            audit_directory = os.path.join(
                self.config.checkpoint_dir, "deep_cfr_strategy_coverage_audit"
            )
            self.strategy_coverage_audit_paths = self.strategy_coverage_audit.write_report(
                audit_directory,
                strategy_reservoirs=self.strategy_buffers,
            )
            if verbose:
                print(
                    "Strategy coverage audit reports: "
                    f"{self.strategy_coverage_audit_paths['summary_json']}",
                    flush=True,
                )
        self._write_metrics(history)
        return history


def train_deep_cfr(
    config: DeepCFRConfig,
    verbose: bool = True,
    show_hardware: bool = True,
) -> Tuple[DeepCFRTrainer, Dict[str, List[Any]]]:
    """Train Deep CFR using the same function-plus-config pattern as PPO/DQN.

    ``config.device=None`` automatically selects CUDA when available. The
    experiment script passes its selected CUDA device explicitly, making GPU
    execution unambiguous in the training report.
    """
    trainer = DeepCFRTrainer(config)
    history = trainer.train(verbose=verbose, show_hardware=show_hardware)
    return trainer, history
