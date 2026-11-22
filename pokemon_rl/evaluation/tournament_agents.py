"""Inference-only adapters for the four Phase 2 tournament policies.

Each adapter receives only the acting player's observation and action mask.
Models are loaded as independent objects for each seat; no adapter receives a
GameState or another player's recurrent/model state.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from pokemon_rl.agents.dqn import DQNetwork, select_action as select_dqn_action
from pokemon_rl.agents.deep_cfr import StrategyNetwork
from pokemon_rl.agents.lstm_ppo import LSTMActorNetwork, MaskedCategorical as LSTMMaskedCategorical
from pokemon_rl.agents.ppo import ActorNetwork, MaskedCategorical as PPOMaskedCategorical
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
ALGORITHM_NAMES = ("DQN", "PPO", "LSTM-PPO", "Deep CFR")

# Relative paths are retained in the tournament config for readability. These
# are the completed Phase 2 checkpoints audited in results/.
DEFAULT_CHECKPOINTS: dict[str, dict[str, str]] = {
    "DQN": {"shared": "results/final_model.pt"},
    "PPO": {"shared": "results/final_ppo_model.pt"},
    "LSTM-PPO": {"shared": "results/final_lstm_ppo_actor.pt"},
    "Deep CFR": {
        "red": "results/final_deep_cfr_red_strategy_masked_ce_iteration_weighted.pt",
        "blue": "results/final_deep_cfr_blue_strategy_masked_ce_iteration_weighted.pt",
    },
}


class AgentAdapter(Protocol):
    """Common game-facing interface; no private game state is passed in."""

    algorithm: str

    def reset(self) -> None: ...

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
    ) -> int: ...


def validate_policy_inputs(
    observation: np.ndarray, action_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation = np.asarray(observation, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=np.float32)
    if observation.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Expected a ({OBSERVATION_DIM},) observation, got {observation.shape}.")
    if action_mask.shape != (ACTION_DIM,):
        raise ValueError(f"Expected a ({ACTION_DIM},) action mask, got {action_mask.shape}.")
    if not np.all((action_mask == 0.0) | (action_mask == 1.0)):
        raise ValueError("Action mask must contain only 0 and 1.")
    legal_actions = np.flatnonzero(action_mask).astype(np.int64, copy=False)
    if legal_actions.size == 0:
        raise ValueError("Cannot select an action from an empty legal-action mask.")
    return observation, action_mask, legal_actions


def _load_state_dict(checkpoint_path: str | Path, device: torch.device) -> dict:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tournament checkpoint does not exist: {path}")
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Could not load tournament checkpoint {path}: {exc}") from exc
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict in {path}, got {type(state_dict).__name__}.")
    return state_dict


class DQNAgent:
    algorithm = "DQN"

    def __init__(self, model: DQNetwork, device: torch.device, checkpoint_path: Path):
        self.model = model.to(device).eval()
        self.device = device
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device) -> "DQNAgent":
        path = Path(checkpoint_path).expanduser().resolve()
        model = DQNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
        model.load_state_dict(_load_state_dict(path, device), strict=True)
        return cls(model, device, path)

    def reset(self) -> None:
        return None

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
    ) -> int:
        observation, action_mask, _ = validate_policy_inputs(observation, action_mask)
        # The existing helper is the canonical masked-greedy DQN selector. It
        # consumes one Python-random draw even at epsilon=0, so preserve the
        # game/deck RNG state around that deterministic call.
        python_rng_state = random.getstate()
        try:
            action = select_dqn_action(
                observation, self.model, action_mask, epsilon=0.0, device=self.device
            )
        finally:
            random.setstate(python_rng_state)
        return int(action)


class PPOAgent:
    algorithm = "PPO"

    def __init__(self, actor: ActorNetwork, device: torch.device, checkpoint_path: Path):
        self.actor = actor.to(device).eval()
        self.device = device
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device) -> "PPOAgent":
        path = Path(checkpoint_path).expanduser().resolve()
        actor = ActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
        actor.load_state_dict(_load_state_dict(path, device), strict=True)
        return cls(actor, device, path)

    def reset(self) -> None:
        return None

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
    ) -> int:
        observation, action_mask, _ = validate_policy_inputs(observation, action_mask)
        observation_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            distribution = PPOMaskedCategorical(self.actor(observation_t), mask_t)
            return int(torch.argmax(distribution.probs).item())


class LSTMPPOAgent:
    algorithm = "LSTM-PPO"

    def __init__(
        self, actor: LSTMActorNetwork, device: torch.device, checkpoint_path: Path
    ):
        self.actor = actor.to(device).eval()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.reset()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: torch.device
    ) -> "LSTMPPOAgent":
        path = Path(checkpoint_path).expanduser().resolve()
        actor = LSTMActorNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
        actor.load_state_dict(_load_state_dict(path, device), strict=True)
        return cls(actor, device, path)

    def reset(self) -> None:
        self._hidden = self.actor.init_hidden(batch_size=1, device=self.device)

    @property
    def hidden_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._hidden is None:
            self.reset()
        assert self._hidden is not None
        return self._hidden

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
    ) -> int:
        observation, action_mask, _ = validate_policy_inputs(observation, action_mask)
        observation_t = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).view(1, 1, OBSERVATION_DIM)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).view(
            1, 1, ACTION_DIM
        )
        with torch.no_grad():
            logits, next_hidden = self.actor(observation_t, self.hidden_state)
            distribution = LSTMMaskedCategorical(logits, mask_t)
            action = int(torch.argmax(distribution.probs.reshape(ACTION_DIM)).item())
        self._hidden = tuple(value.detach() for value in next_hidden)
        return action


class DeepCFRStrategyAgent:
    algorithm = "Deep CFR"

    def __init__(
        self,
        strategy_network: StrategyNetwork,
        device: torch.device,
        checkpoint_path: Path,
        player_idx: int,
    ):
        if player_idx not in (0, 1):
            raise ValueError("player_idx must be 0 (Red) or 1 (Blue).")
        self.player_idx = player_idx
        self.strategy_network = strategy_network.to(device).eval()
        self.strategy_network.device = device
        self.device = device
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: torch.device, player_idx: int
    ) -> "DeepCFRStrategyAgent":
        path = Path(checkpoint_path).expanduser().resolve()
        network = StrategyNetwork(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
        network.device = device
        network.load_state_dict(_load_state_dict(path, device), strict=True)
        return cls(network, device, path, player_idx)

    def reset(self) -> None:
        return None

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
    ) -> int:
        observation, action_mask, legal_actions = validate_policy_inputs(observation, action_mask)
        probabilities = np.asarray(
            self.strategy_network.get_strategy(observation, action_mask), dtype=np.float64
        )
        if probabilities.shape != (ACTION_DIM,) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Deep CFR StrategyNetwork returned invalid probabilities.")
        if np.any(probabilities[~action_mask.astype(bool)] != 0.0):
            raise RuntimeError("Deep CFR StrategyNetwork assigned mass to an illegal action.")
        legal_probabilities = probabilities[legal_actions]
        if np.any(legal_probabilities < 0.0):
            raise RuntimeError("Deep CFR StrategyNetwork returned a negative legal probability.")
        mass = float(legal_probabilities.sum())
        if mass <= 0.0:
            raise RuntimeError("Deep CFR StrategyNetwork has no legal probability mass.")
        legal_probabilities = legal_probabilities / mass
        return int(rng.choice(legal_actions, p=legal_probabilities))


def default_checkpoint_path(algorithm: str, player_idx: int) -> Path:
    """Resolve the audited final checkpoint for the algorithm and seat."""
    if algorithm not in DEFAULT_CHECKPOINTS or player_idx not in (0, 1):
        raise ValueError(f"Unknown algorithm/seat combination: {algorithm!r}, {player_idx}.")
    role = "red" if player_idx == 0 else "blue"
    relative_path = DEFAULT_CHECKPOINTS[algorithm].get("shared") or DEFAULT_CHECKPOINTS[algorithm][role]
    return PROJECT_ROOT / relative_path


def load_agent(algorithm: str, player_idx: int, device: torch.device) -> AgentAdapter:
    """Load a fresh adapter/model instance for one tournament seat."""
    checkpoint = default_checkpoint_path(algorithm, player_idx)
    if algorithm == "DQN":
        return DQNAgent.from_checkpoint(checkpoint, device)
    if algorithm == "PPO":
        return PPOAgent.from_checkpoint(checkpoint, device)
    if algorithm == "LSTM-PPO":
        return LSTMPPOAgent.from_checkpoint(checkpoint, device)
    if algorithm == "Deep CFR":
        return DeepCFRStrategyAgent.from_checkpoint(checkpoint, device, player_idx)
    raise ValueError(f"Unknown tournament algorithm: {algorithm!r}")

