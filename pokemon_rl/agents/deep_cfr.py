import os
import sys
import copy
import random
import hashlib
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Any, Mapping, Optional

from pokemon_rl.cards.pokemon import PokemonCard
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.player import Player, Deck
from pokemon_rl.game import GameEngine, Action, GameState
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM, get_phase2_information_set_observation

class RegretNetwork(nn.Module):
    """PPO-like Regret Network for Deep CFR in Mini Pokémon TCG.
    
    Maps the Phase 2 50-dimensional information set observation to 70 predicted
    cumulative regret values. Supports both single inputs (50,) and batched
    inputs (B, 50).
    """
    def __init__(self, state_dim: int = OBSERVATION_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 128):
        super(RegretNetwork, self).__init__()
        if state_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"Phase 2 RegretNetwork requires state_dim={OBSERVATION_DIM} and "
                f"action_dim={ACTION_DIM}; got state_dim={state_dim}, action_dim={action_dim}."
            )
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        if x.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 RegretNetwork expects final input dimension {OBSERVATION_DIM}, "
                f"got {x.shape[-1]}."
            )
        return self.net(x)


class StrategyNetwork(nn.Module):
    """Strategy Network for Deep CFR in Mini Pokémon TCG.
    
    Predicts the converged average strategy (probabilities) over 70 discrete actions.
    Enforces action masks so illegal actions receive zero probability.
    """
    def __init__(self, state_dim: int = OBSERVATION_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 128):
        super(StrategyNetwork, self).__init__()
        if state_dim != OBSERVATION_DIM or action_dim != ACTION_DIM:
            raise ValueError(
                f"Phase 2 StrategyNetwork requires state_dim={OBSERVATION_DIM} and "
                f"action_dim={ACTION_DIM}; got state_dim={state_dim}, action_dim={action_dim}."
            )
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        if x.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Phase 2 StrategyNetwork expects final input dimension {OBSERVATION_DIM}, "
                f"got {x.shape[-1]}."
            )
        return self.net(x)

    def get_strategy(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Returns the re-normalized legal strategy probability distribution.
        
        Supports both single observations (50,) and batched observations (B, 50).
        """
        obs = np.asarray(obs, dtype=np.float32)
        action_mask = np.asarray(action_mask, dtype=np.float32)
        if obs.ndim not in (1, 2) or obs.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                f"Strategy observations must have shape ({OBSERVATION_DIM},) or (B, {OBSERVATION_DIM}), "
                f"got {obs.shape}."
            )
        if action_mask.shape != (*obs.shape[:-1], ACTION_DIM):
            raise ValueError(
                f"Strategy action masks must match observation batches and end in {ACTION_DIM}; "
                f"got {action_mask.shape}."
            )
        if not np.all((action_mask == 0.0) | (action_mask == 1.0)):
            raise ValueError("Strategy action mask must be binary.")
        if np.any(action_mask.sum(axis=-1) == 0.0):
            raise ValueError("Strategy action mask must contain at least one legal action per state.")

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
        
        is_single = len(obs_t.shape) == 1
        if is_single:
            obs_t = obs_t.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0)
            
        with torch.no_grad():
            logits = self.forward(obs_t)
            
            # A finite lower bound keeps single-action forced-promotion rows
            # numerically stable while giving every illegal action probability 0.
            masked_logits = logits.masked_fill(mask_t == 0.0, torch.finfo(logits.dtype).min)
            
            # Compute raw Softmax probabilities
            probs = torch.softmax(masked_logits, dim=-1)
            
            # Force exact zeros for masked options and re-normalize to handle any precision limits
            probs = probs * mask_t
            sums = probs.sum(dim=-1, keepdim=True)
            
            if not torch.all(torch.isfinite(sums)) or torch.any(sums == 0.0):
                raise RuntimeError("Masked strategy normalization became invalid.")
            probs = probs / sums
                    
        res = probs.cpu().numpy()
        if is_single:
            res = res[0]
        return res


def regret_matching(regrets: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Computes a regret-matching probability distribution over legal actions.
    
    If all positive legal regrets sum to zero, falls back to a uniform distribution
    over the legal actions. Illegal actions are guaranteed to have exactly 0 probability.
    """
    regrets = np.asarray(regrets, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=np.float32)
    if regrets.shape != (ACTION_DIM,) or action_mask.shape != (ACTION_DIM,):
        raise ValueError(
            f"Phase 2 regret matching requires regret and mask vectors of shape ({ACTION_DIM},); "
            f"got {regrets.shape} and {action_mask.shape}."
        )
    if not np.all((action_mask == 0.0) | (action_mask == 1.0)):
        raise ValueError("Regret-matching action mask must be binary.")
    if np.sum(action_mask) == 0.0:
        raise ValueError("Regret-matching action mask has no legal actions.")
    
    # Calculate positive regrets: max(regret, 0)
    pos_regret = np.maximum(regrets, 0.0)
    
    # Enforce action masking
    legal_pos_regrets = pos_regret * action_mask
    
    sum_pos_regrets = np.sum(legal_pos_regrets)
    
    if sum_pos_regrets > 0.0:
        strategy = legal_pos_regrets / sum_pos_regrets
    else:
        # Uniform fallback over legal actions
        legal_count = np.sum(action_mask)
        strategy = action_mask / legal_count
            
    return strategy


class ReservoirBuffer:
    """Generic memory buffer utilizing true Reservoir Sampling (Algorithm R) for Deep CFR.

    Regret samples are ``(observation, target)`` pairs.  Strategy samples also
    retain the exact legal-action mask from target generation as
    ``(observation, target, action_mask, iteration_weight)``.  The target
    remains a normalized legal probability distribution; ``iteration_weight``
    is a separate positive scalar used only by the StrategyNetwork's weighted
    cross-entropy objective.  A target alone cannot identify every legal action
    because regret matching may correctly assign a legal action probability
    zero.
    """
    def __init__(self, capacity: int, seed: Optional[int] = None,
                 state_dim: Optional[int] = None, target_dim: Optional[int] = None,
                 mask_dim: Optional[int] = None, sample_weight: bool = False):
        if capacity <= 0:
            raise ValueError("Reservoir capacity must be positive.")
        if (state_dim is None) != (target_dim is None):
            raise ValueError("Reservoir state_dim and target_dim must be provided together.")
        if mask_dim is not None and state_dim is None:
            raise ValueError("A masked Deep CFR reservoir requires state_dim and target_dim.")
        if sample_weight and mask_dim is None:
            raise ValueError("A weighted Deep CFR reservoir requires an action mask.")
        self.capacity = capacity
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.mask_dim = mask_dim
        self.sample_weight = sample_weight
        self.buffer = []
        self.num_samples_seen = 0
        self.rng = random.Random(seed)

    def add(self, sample: Any) -> Optional[int]:
        """Append with Algorithm R and return the retained slot, if any.

        Callers that ignore the return value retain the original reservoir
        behavior.  The optional slot is observational metadata for the
        StrategyCoverageAudit only; it does not influence validation, random
        replacement, or the stored sample.
        """
        if self.state_dim is not None:
            sample_size = 4 if self.sample_weight else 3 if self.mask_dim is not None else 2
            if not isinstance(sample, tuple) or len(sample) != sample_size:
                description = (
                    "(information_state, target, action_mask, iteration_weight)"
                    if self.sample_weight
                    else "(information_state, target, action_mask)"
                    if self.mask_dim is not None
                    else "(information_state, target)"
                )
                raise ValueError(f"Configured Deep CFR reservoirs require {description} samples.")
            observation = np.asarray(sample[0], dtype=np.float32)
            target = np.asarray(sample[1], dtype=np.float32)
            if observation.shape != (self.state_dim,) or target.shape != (self.target_dim,):
                raise ValueError(
                    f"Deep CFR reservoir expects sample shapes ({self.state_dim},) and ({self.target_dim},), "
                    f"got {observation.shape} and {target.shape}."
                )
            if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(target)):
                raise ValueError("Deep CFR reservoir samples must be finite.")
            if self.mask_dim is None:
                sample = (observation.copy(), target.copy())
            else:
                action_mask = np.asarray(sample[2], dtype=np.float32)
                if action_mask.shape != (self.mask_dim,):
                    raise ValueError(
                        f"Deep CFR strategy masks must have shape ({self.mask_dim},), "
                        f"got {action_mask.shape}."
                    )
                if not np.all((action_mask == 0.0) | (action_mask == 1.0)):
                    raise ValueError("Deep CFR strategy masks must be binary.")
                if np.sum(action_mask) == 0.0:
                    raise ValueError("Deep CFR strategy masks must contain a legal action.")
                if np.any(target < 0.0) or np.any(target[action_mask == 0.0] != 0.0):
                    raise ValueError("Deep CFR strategy targets must have zero mass on illegal actions.")
                if not np.isclose(float(target[action_mask == 1.0].sum()), 1.0, atol=1e-6):
                    raise ValueError("Deep CFR strategy targets must normalize over legal actions.")
                if self.sample_weight:
                    weight_array = np.asarray(sample[3], dtype=np.float32)
                    if weight_array.shape != ():
                        raise ValueError("Deep CFR strategy sample weights must be scalar values.")
                    iteration_weight = float(weight_array)
                    if not np.isfinite(iteration_weight) or iteration_weight <= 0.0:
                        raise ValueError("Deep CFR strategy sample weights must be finite and positive.")
                    sample = (
                        observation.copy(), target.copy(), action_mask.copy(), iteration_weight,
                    )
                else:
                    sample = (observation.copy(), target.copy(), action_mask.copy())
        retained_slot: Optional[int] = None
        if len(self.buffer) < self.capacity:
            retained_slot = len(self.buffer)
            self.buffer.append(sample)
        else:
            idx = self.rng.randint(0, self.num_samples_seen)
            if idx < self.capacity:
                self.buffer[idx] = sample
                retained_slot = idx
        self.num_samples_seen += 1
        return retained_slot

    def sample(self, batch_size: int) -> List[Any]:
        """Uniformly samples batch_size items without modifying the buffer.
        
        Uses choices (with replacement) to support sampling even when buffer size < batch_size.
        """
        if len(self.buffer) == 0:
            return []
        return self.rng.choices(self.buffer, k=batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Resets the reservoir buffer."""
        self.buffer = []
        self.num_samples_seen = 0


class _StrategyTargetRunningStats:
    """Compact Welford statistics for one observation-plus-mask key."""

    def __init__(self, target: np.ndarray):
        self.count = 1
        self.mean = np.asarray(target, dtype=np.float32).copy()
        self.m2 = np.zeros_like(self.mean)
        # This is intentionally a scalar, rather than all historical targets:
        # it gives a cheap disagreement signal without pairwise comparisons.
        self.max_linf_deviation_from_prior_mean = 0.0

    def add(self, target: np.ndarray) -> None:
        target_array = np.asarray(target, dtype=np.float32)
        previous_mean = self.mean.copy()
        self.max_linf_deviation_from_prior_mean = max(
            self.max_linf_deviation_from_prior_mean,
            float(np.max(np.abs(target_array - previous_mean))),
        )
        self.count += 1
        delta = target_array - previous_mean
        self.mean += delta / self.count
        self.m2 += delta * (target_array - self.mean)

    def variance(self) -> np.ndarray:
        return self.m2 / self.count


class StrategyCoverageAudit:
    """Passive coverage accounting for Deep CFR strategy-memory samples.

    This audit is intentionally independent of CFR decisions.  It observes
    metadata *after* a strategy reservoir has made its normal Algorithm R
    replacement decision, retaining only deterministic hashes, counters,
    running target statistics, and a compact shadow of final reservoir slots.
    Observation hashes are diagnostic fingerprints, not canonical information
    set identifiers.
    """

    _PLAYER_NAMES = {0: "red", 1: "blue"}
    _PROMOTION_ACTIONS = frozenset((68, 69))

    def __init__(self, reservoir_capacity: int):
        if reservoir_capacity <= 0:
            raise ValueError("Strategy coverage audit reservoir capacity must be positive.")
        self.reservoir_capacity = int(reservoir_capacity)
        self._player_data = {player: self._new_player_data() for player in self._PLAYER_NAMES}
        # Each value is (depth, observation_hash, mask_hash, legal_action_count)
        # or None when no strategy sample has ever occupied that slot.
        self._reservoir_slots = {
            player: [None] * self.reservoir_capacity for player in self._PLAYER_NAMES
        }

    @staticmethod
    def _stable_array_hash(values: np.ndarray, dtype: np.dtype) -> bytes:
        """Hash array contents with explicit byte order and shape metadata."""
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        shape_bytes = np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes()
        return hashlib.blake2b(shape_bytes + array.tobytes(order="C"), digest_size=16).digest()

    @classmethod
    def observation_hash(cls, observation: np.ndarray) -> bytes:
        """Return a stable fingerprint for one 50-D encoded observation."""
        return cls._stable_array_hash(observation, np.dtype("<f4"))

    @classmethod
    def action_mask_hash(cls, action_mask: np.ndarray) -> bytes:
        """Return a stable fingerprint for one binary 70-action mask."""
        return cls._stable_array_hash(action_mask, np.dtype("u1"))

    @staticmethod
    def _new_player_data() -> Dict[str, Any]:
        return {
            "stream_total": 0,
            "stream_by_depth": Counter(),
            "stream_by_legal_action_count": Counter(),
            # Keyed by (depth, observation_hash, action_mask_hash). Player is
            # deliberately kept outside this key by the separate dictionaries.
            "stream_key_counts": Counter(),
            "observation_counts": Counter(),
            # Keyed by (observation_hash, action_mask_hash), aggregating across
            # depths so repeated target disagreement is directly measurable.
            "target_stats": {},
            "mask_counts": Counter(),
            "mask_patterns": {},
        }

    @staticmethod
    def _validate_sample_metadata(
        player: int,
        depth: int,
        observation: np.ndarray,
        target: np.ndarray,
        action_mask: np.ndarray,
    ) -> None:
        if player not in (0, 1):
            raise ValueError("Strategy coverage audit player must be Red (0) or Blue (1).")
        if not isinstance(depth, (int, np.integer)) or depth < 0:
            raise ValueError("Strategy coverage audit depth must be a non-negative integer.")
        if np.asarray(observation).shape != (OBSERVATION_DIM,):
            raise ValueError("Strategy coverage audit observations must have the Phase 2 state shape.")
        if np.asarray(target).shape != (ACTION_DIM,) or np.asarray(action_mask).shape != (ACTION_DIM,):
            raise ValueError("Strategy coverage audit targets and masks must have the Phase 2 action shape.")

    def record_strategy_sample(
        self,
        *,
        player: int,
        depth: int,
        observation: np.ndarray,
        target: np.ndarray,
        action_mask: np.ndarray,
        retained_slot: Optional[int],
    ) -> None:
        """Observe one already-validated strategy insertion without mutation.

        ``retained_slot`` is returned by the real strategy reservoir after its
        Algorithm R decision.  ``None`` means the item was not retained.  The
        audit never invokes the reservoir RNG and never changes its contents.
        """
        self._validate_sample_metadata(player, depth, observation, target, action_mask)
        if retained_slot is not None and not 0 <= retained_slot < self.reservoir_capacity:
            raise ValueError("Strategy coverage audit received an invalid reservoir slot.")

        observation_hash = self.observation_hash(observation)
        mask_hash = self.action_mask_hash(action_mask)
        legal_indices = tuple(int(index) for index in np.flatnonzero(action_mask))
        legal_action_count = len(legal_indices)
        data = self._player_data[player]

        data["stream_total"] += 1
        data["stream_by_depth"][int(depth)] += 1
        data["stream_by_legal_action_count"][legal_action_count] += 1
        data["stream_key_counts"][(int(depth), observation_hash, mask_hash)] += 1
        data["observation_counts"][observation_hash] += 1
        data["mask_counts"][mask_hash] += 1
        data["mask_patterns"].setdefault(mask_hash, legal_indices)

        observation_mask_key = (observation_hash, mask_hash)
        target_stats = data["target_stats"].get(observation_mask_key)
        if target_stats is None:
            data["target_stats"][observation_mask_key] = _StrategyTargetRunningStats(target)
        else:
            target_stats.add(target)

        if retained_slot is not None:
            self._reservoir_slots[player][retained_slot] = (
                int(depth), observation_hash, mask_hash, legal_action_count,
            )

    @staticmethod
    def _histogram(counter: Counter) -> Dict[str, int]:
        return {str(key): int(value) for key, value in sorted(counter.items())}

    @staticmethod
    def _duplicate_frequency_distribution(counts: Counter) -> Dict[str, int]:
        return StrategyCoverageAudit._histogram(Counter(counts.values()))

    @classmethod
    def _mask_report(
        cls,
        mask_counts: Counter,
        mask_patterns: Dict[bytes, Tuple[int, ...]],
    ) -> Dict[str, Any]:
        legal_action_counts = Counter()
        forced_promotion_masks = 0
        single_action_masks = 0
        two_action_masks = 0
        for mask_hash, occurrences in mask_counts.items():
            legal_indices = mask_patterns[mask_hash]
            legal_count = len(legal_indices)
            legal_action_counts[legal_count] += occurrences
            if legal_count == 1:
                single_action_masks += occurrences
            if legal_count == 2:
                two_action_masks += occurrences
            if legal_indices and set(legal_indices).issubset(cls._PROMOTION_ACTIONS):
                forced_promotion_masks += occurrences
        top_masks = []
        for mask_hash, occurrences in mask_counts.most_common(20):
            legal_indices = mask_patterns[mask_hash]
            top_masks.append({
                "action_mask_hash": mask_hash.hex(),
                "occurrences": int(occurrences),
                "legal_action_count": len(legal_indices),
                "legal_action_indices": list(legal_indices),
            })
        return {
            "unique_mask_patterns": len(mask_counts),
            "legal_action_count_histogram": cls._histogram(legal_action_counts),
            "forced_promotion_mask_occurrences": int(forced_promotion_masks),
            "single_action_mask_occurrences": int(single_action_masks),
            "two_action_mask_occurrences": int(two_action_masks),
            "most_frequent_mask_patterns": top_masks,
        }

    @staticmethod
    def _target_disagreement_report(target_stats: Dict[Tuple[bytes, bytes], _StrategyTargetRunningStats]) -> Dict[str, Any]:
        repeated = [stats for stats in target_stats.values() if stats.count > 1]
        if not repeated:
            return {
                "repeated_observation_mask_keys": 0,
                "keys_with_nonzero_target_variation": 0,
                "mean_target_variance_across_repeated_keys": 0.0,
                "maximum_target_variance": 0.0,
                "maximum_observed_linf_deviation_from_prior_mean": 0.0,
            }
        key_variances = [float(np.mean(stats.variance())) for stats in repeated]
        max_variance = max(float(np.max(stats.variance())) for stats in repeated)
        nonzero = sum(bool(np.any(stats.variance() > 0.0)) for stats in repeated)
        return {
            "repeated_observation_mask_keys": len(repeated),
            "keys_with_nonzero_target_variation": int(nonzero),
            "mean_target_variance_across_repeated_keys": float(np.mean(key_variances)),
            "maximum_target_variance": float(max_variance),
            "maximum_observed_linf_deviation_from_prior_mean": float(max(
                stats.max_linf_deviation_from_prior_mean for stats in repeated
            )),
        }

    def _build_player_report(self, player: int) -> Dict[str, Any]:
        data = self._player_data[player]
        occupied_slots = [slot for slot in self._reservoir_slots[player] if slot is not None]
        reservoir_by_depth = Counter(slot[0] for slot in occupied_slots)
        reservoir_by_legal_count = Counter(slot[3] for slot in occupied_slots)
        reservoir_observation_hashes = {slot[1] for slot in occupied_slots}
        reservoir_observation_mask_hashes = {(slot[1], slot[2]) for slot in occupied_slots}
        reservoir_mask_counts = Counter(slot[2] for slot in occupied_slots)

        inserted_observations_by_depth: Dict[int, set] = {}
        inserted_observation_masks_by_depth: Dict[int, set] = {}
        for depth, observation_hash, mask_hash in data["stream_key_counts"]:
            inserted_observations_by_depth.setdefault(depth, set()).add(observation_hash)
            inserted_observation_masks_by_depth.setdefault(depth, set()).add((observation_hash, mask_hash))
        reservoir_observations_by_depth: Dict[int, set] = {}
        reservoir_observation_masks_by_depth: Dict[int, set] = {}
        for depth, observation_hash, mask_hash, _ in occupied_slots:
            reservoir_observations_by_depth.setdefault(depth, set()).add(observation_hash)
            reservoir_observation_masks_by_depth.setdefault(depth, set()).add((observation_hash, mask_hash))

        depth_coverage: Dict[str, Dict[str, Any]] = {}
        all_depths = set(data["stream_by_depth"]) | set(reservoir_by_depth)
        for depth in sorted(all_depths):
            inserted_count = int(data["stream_by_depth"][depth])
            retained_count = int(reservoir_by_depth[depth])
            inserted_observations = inserted_observations_by_depth.get(depth, set())
            retained_observations = reservoir_observations_by_depth.get(depth, set())
            inserted_observation_masks = inserted_observation_masks_by_depth.get(depth, set())
            retained_observation_masks = reservoir_observation_masks_by_depth.get(depth, set())
            depth_coverage[str(depth)] = {
                "inserted_samples": inserted_count,
                "final_reservoir_samples": retained_count,
                "retained_sample_fraction_percent": (
                    100.0 * retained_count / inserted_count if inserted_count else 0.0
                ),
                "inserted_unique_observation_hashes": len(inserted_observations),
                "surviving_unique_observation_hashes": len(retained_observations),
                "observation_hash_coverage_percent": (
                    100.0 * len(retained_observations) / len(inserted_observations)
                    if inserted_observations else 0.0
                ),
                "inserted_unique_observation_mask_hashes": len(inserted_observation_masks),
                "surviving_unique_observation_mask_hashes": len(retained_observation_masks),
                "observation_mask_hash_coverage_percent": (
                    100.0 * len(retained_observation_masks) / len(inserted_observation_masks)
                    if inserted_observation_masks else 0.0
                ),
            }

        stream_observation_mask_counts = Counter(
            stats.count for stats in data["target_stats"].values()
        )
        return {
            "stream": {
                "total_strategy_samples_inserted": int(data["stream_total"]),
                "samples_by_depth": self._histogram(data["stream_by_depth"]),
                "samples_by_legal_action_count": self._histogram(data["stream_by_legal_action_count"]),
                "unique_observation_hashes": len(data["observation_counts"]),
                "unique_observation_mask_hashes": len(data["target_stats"]),
                "observation_duplicate_frequency_distribution": self._duplicate_frequency_distribution(
                    data["observation_counts"]
                ),
                "observation_mask_duplicate_frequency_distribution": self._histogram(
                    stream_observation_mask_counts
                ),
                "mask_distribution": self._mask_report(
                    data["mask_counts"], data["mask_patterns"]
                ),
            },
            "final_reservoir": {
                "size": len(occupied_slots),
                "samples_by_depth": self._histogram(reservoir_by_depth),
                "samples_by_legal_action_count": self._histogram(reservoir_by_legal_count),
                "unique_observation_hashes": len(reservoir_observation_hashes),
                "unique_observation_mask_hashes": len(reservoir_observation_mask_hashes),
                "mask_distribution": self._mask_report(
                    reservoir_mask_counts, data["mask_patterns"]
                ),
            },
            "target_disagreement": self._target_disagreement_report(data["target_stats"]),
            "depth_coverage": depth_coverage,
        }

    def summary(self) -> Dict[str, Any]:
        """Return machine-readable, Red/Blue-separated aggregate diagnostics."""
        return {
            "schema_version": 1,
            "terminology_note": (
                "Observation hashes and observation-plus-mask hashes are deterministic "
                "fingerprints of the 50-D encoder and 70-D legal mask. They are not "
                "guaranteed canonical information-set identifiers."
            ),
            "reservoir_capacity": self.reservoir_capacity,
            "players": {
                self._PLAYER_NAMES[player]: self._build_player_report(player)
                for player in self._PLAYER_NAMES
            },
        }

    def _write_repeated_target_details(self, path: str) -> None:
        """Write per-repeated-key target means/variances without pairwise work."""
        with open(path, "w", encoding="utf-8") as detail_file:
            for player, player_name in self._PLAYER_NAMES.items():
                data = self._player_data[player]
                for (observation_hash, mask_hash), stats in data["target_stats"].items():
                    if stats.count <= 1:
                        continue
                    record = {
                        "player": player_name,
                        "observation_hash": observation_hash.hex(),
                        "action_mask_hash": mask_hash.hex(),
                        "legal_action_indices": list(data["mask_patterns"][mask_hash]),
                        "observation_count": int(stats.count),
                        "running_mean_target": [float(value) for value in stats.mean],
                        "running_target_variance": [float(value) for value in stats.variance()],
                        "max_observed_linf_deviation_from_prior_mean": float(
                            stats.max_linf_deviation_from_prior_mean
                        ),
                    }
                    detail_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _write_final_reservoir_index(
        self,
        path: str,
        strategy_reservoirs: Optional[Mapping[int, "ReservoirBuffer"]] = None,
    ) -> None:
        """Write compact final-reservoir fingerprints for evaluation coverage.

        This deliberately emits only hashes and static metadata, never raw
        observations, masks, targets, or training samples.  Evaluation
        coverage can therefore test exact final-reservoir membership without
        reconstructing or modifying a checkpoint.

        When the trainer provides its reservoirs, records are derived from the
        actual samples that survived Algorithm R.  The audit's slot metadata is
        retained solely for traversal depth and is checked against each final
        sample before it is emitted.
        """
        with open(path, "w", encoding="utf-8") as index_file:
            for player, player_name in self._PLAYER_NAMES.items():
                if strategy_reservoirs is None:
                    final_slots = enumerate(self._reservoir_slots[player])
                else:
                    final_slots = enumerate(strategy_reservoirs[player].buffer)

                for reservoir_slot, sample_or_metadata in final_slots:
                    if strategy_reservoirs is None:
                        metadata = sample_or_metadata
                        if metadata is None:
                            continue
                        depth, observation_hash, mask_hash, legal_action_count = metadata
                    else:
                        observation, _target, action_mask, _iteration_weight = sample_or_metadata
                        observation_hash = self.observation_hash(observation)
                        mask_hash = self.action_mask_hash(action_mask)
                        legal_action_count = int(np.count_nonzero(action_mask))
                        metadata = self._reservoir_slots[player][reservoir_slot]
                        if metadata is None:
                            raise RuntimeError(
                                "Strategy coverage audit is missing metadata for "
                                f"the final {player_name} reservoir slot {reservoir_slot}."
                            )
                        (
                            depth,
                            recorded_observation_hash,
                            recorded_mask_hash,
                            recorded_legal_action_count,
                        ) = metadata
                        if (
                            recorded_observation_hash != observation_hash
                            or recorded_mask_hash != mask_hash
                            or recorded_legal_action_count != legal_action_count
                        ):
                            raise RuntimeError(
                                "Strategy coverage audit metadata does not match "
                                f"the final {player_name} reservoir slot {reservoir_slot}."
                            )
                    record = {
                        "player": player_name,
                        "reservoir_slot": reservoir_slot,
                        "traversal_depth": depth,
                        "observation_hash": observation_hash.hex(),
                        "action_mask_hash": mask_hash.hex(),
                        "legal_action_count": legal_action_count,
                    }
                    index_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    @staticmethod
    def _human_summary(summary: Dict[str, Any]) -> str:
        lines = [
            "DEEP CFR STRATEGY COVERAGE AUDIT",
            "=" * 70,
            summary["terminology_note"],
            f"Reservoir capacity per player: {summary['reservoir_capacity']}",
        ]
        for player_name, player_report in summary["players"].items():
            stream = player_report["stream"]
            reservoir = player_report["final_reservoir"]
            disagreement = player_report["target_disagreement"]
            lines.extend((
                "",
                player_name.upper(),
                f"  Stream samples: {stream['total_strategy_samples_inserted']}",
                f"  Stream unique observation hashes: {stream['unique_observation_hashes']}",
                f"  Stream unique observation+mask hashes: {stream['unique_observation_mask_hashes']}",
                f"  Final reservoir samples: {reservoir['size']}",
                f"  Final reservoir unique observation hashes: {reservoir['unique_observation_hashes']}",
                f"  Final reservoir unique observation+mask hashes: {reservoir['unique_observation_mask_hashes']}",
                f"  Repeated observation+mask keys: {disagreement['repeated_observation_mask_keys']}",
                f"  Repeated keys with target variation: {disagreement['keys_with_nonzero_target_variation']}",
                f"  Mean / max target variance: "
                f"{disagreement['mean_target_variance_across_repeated_keys']:.8f} / "
                f"{disagreement['maximum_target_variance']:.8f}",
                f"  Inserted samples by depth: {stream['samples_by_depth']}",
                f"  Final reservoir samples by depth: {reservoir['samples_by_depth']}",
            ))
        return "\n".join(lines) + "\n"

    def write_report(
        self,
        output_dir: str,
        strategy_reservoirs: Optional[Mapping[int, "ReservoirBuffer"]] = None,
    ) -> Dict[str, str]:
        """Write separate machine-readable and concise human-readable reports."""
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "strategy_coverage_summary.json")
        text_path = os.path.join(output_dir, "strategy_coverage_summary.txt")
        repeated_targets_path = os.path.join(
            output_dir, "strategy_coverage_repeated_observation_mask_targets.jsonl"
        )
        final_reservoir_index_path = os.path.join(
            output_dir, "strategy_coverage_final_reservoir_index.jsonl"
        )
        summary = self.summary()
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(self._human_summary(summary))
        self._write_repeated_target_details(repeated_targets_path)
        self._write_final_reservoir_index(
            final_reservoir_index_path,
            strategy_reservoirs=strategy_reservoirs,
        )
        return {
            "summary_json": summary_path,
            "summary_text": text_path,
            "repeated_targets_jsonl": repeated_targets_path,
            "final_reservoir_index_jsonl": final_reservoir_index_path,
        }


def create_game_state_snapshot(game_state: GameState) -> GameState:
    """Deep-copy every Phase 2 game field before branching a traversal.

    ``GameState`` owns active and Bench Pokémon, attached typed Energy,
    hands/decks/trash, per-turn flags, forced-promotion state, knockout
    counts, winner state, and turn order.  A deep copy keeps all of those
    mutable values isolated between MCCFR siblings.
    """
    return copy.deepcopy(game_state)


def restore_game_state_snapshot(snapshot: GameState) -> GameState:
    """Return an independent Phase 2 state restored from a deep snapshot."""
    return copy.deepcopy(snapshot)


def get_information_set_observation(game_state: GameState, player_idx: int) -> np.ndarray:
    """Return the exact Phase 2, player-relative information-state vector.

    This delegates to the shared pure encoder rather than the Gymnasium
    wrapper. It exposes the viewing player's permitted hand slots and public
    boards, but encodes the opponent's hand and deck as sizes only.
    """
    return get_phase2_information_set_observation(game_state, player_idx)


def _game_action_to_index(action: Action) -> int:
    """Encode one concrete, representable ``GameEngine`` action as 0--69.

    Hand slots are checked before their action groups are calculated, so a
    ninth-hand-card action can never alias to a different Phase 2 group.
    ``GameEngine`` itself only offers slots 0--7, but this explicit guard
    preserves that invariant for all direct Deep CFR callers.
    """
    if action.action_type == "end_turn":
        return 0
    if action.action_type == "attack" and action.attack_idx in (0, 1):
        return 1 + action.attack_idx
    if action.action_type == "retreat":
        return 3

    if action.card_idx is not None and not 0 <= action.card_idx < GameEngine.VISIBLE_HAND_SLOTS:
        raise ValueError(f"Hand slot {action.card_idx} is outside the fixed visible range 0--7.")
    if action.action_type == "play_pokemon" and action.bench_slot in (0, 1) and action.card_idx is not None:
        return (4 if action.bench_slot == 0 else 12) + action.card_idx
    if action.action_type == "attach_energy" and action.card_idx is not None:
        if action.target == "active":
            return 20 + action.card_idx
        if action.target == "bench" and action.bench_slot in (0, 1):
            return (28 if action.bench_slot == 0 else 36) + action.card_idx
    if action.action_type == "play_potion" and action.card_idx is not None:
        if action.target == "active":
            return 44 + action.card_idx
        if action.target == "bench" and action.bench_slot in (0, 1):
            return (52 if action.bench_slot == 0 else 60) + action.card_idx
    if action.action_type == "promote" and action.bench_slot in (0, 1):
        return 68 + action.bench_slot
    raise ValueError(f"Cannot map GameEngine action to Phase 2 Deep CFR index: {action!r}")


def get_legal_action_mask(game_state: GameState, player_idx: int) -> np.ndarray:
    """Return the exact Phase 2 70-action mask for ``player_idx``.

    The conversion is performed directly from ``GameEngine`` actions, keeping
    two-player traversal independent of the Gymnasium Random-Blue wrapper.
    """
    if player_idx not in (0, 1):
        raise ValueError("player_idx must be 0 or 1.")
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    if game_state.game_over or game_state.current_player_idx != player_idx:
        return mask
    for action in GameEngine.get_legal_actions(game_state):
        action_index = _game_action_to_index(action)
        mask[action_index] = 1.0
                
    return mask


def train_regret_network(
    network: RegretNetwork,
    optimizer: torch.optim.Optimizer,
    batch: List[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """Trains the RegretNetwork on a batch of (obs, regret_target) tuples using MSE loss."""
    if len(batch) == 0:
        return 0.0
        
    network.train()
    
    states_list = [np.asarray(item[0], dtype=np.float32) for item in batch]
    targets_list = [np.asarray(item[1], dtype=np.float32) for item in batch]
    if any(state.shape != (OBSERVATION_DIM,) for state in states_list):
        raise ValueError(f"Regret samples must use Phase 2 states shaped ({OBSERVATION_DIM},).")
    if any(target.shape != (ACTION_DIM,) for target in targets_list):
        raise ValueError(f"Regret targets must use the Phase 2 {ACTION_DIM}-action shape.")
    
    states_t = torch.tensor(np.array(states_list), dtype=torch.float32, device=network.device)
    targets_t = torch.tensor(np.array(targets_list), dtype=torch.float32, device=network.device)
    
    predictions = network(states_t)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(predictions, targets_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def masked_strategy_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    action_masks: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return iteration-weighted legal-action cross-entropy for strategy targets.

    ``action_masks`` come from the exact information state at which the target
    was generated.  Masking logits before ``log_softmax`` prevents illegal
    outputs from affecting legal probabilities or receiving gradients.  The
    target is detached and normalized only over that legal support; this also
    handles one-action and forced-promotion states without changing the target
    probability semantics.  When provided, ``sample_weights`` holds one
    positive scalar per strategy sample (the outer CFR iteration) and produces
    ``sum(weight * per_sample_CE) / sum(weight)``.  The target distribution is
    never scaled by that weight.
    """
    if logits.ndim != 2 or logits.shape[-1] != ACTION_DIM:
        raise ValueError(f"Strategy logits must have shape (B, {ACTION_DIM}), got {tuple(logits.shape)}.")
    if targets.shape != logits.shape or action_masks.shape != logits.shape:
        raise ValueError("Strategy logits, targets, and action masks must have identical batch shapes.")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Strategy logits and targets must be finite.")

    masks = action_masks.to(device=logits.device, dtype=logits.dtype)
    detached_targets = targets.detach().to(device=logits.device, dtype=logits.dtype)
    if not torch.all((masks == 0.0) | (masks == 1.0)):
        raise ValueError("Strategy action masks must be binary.")
    if torch.any(masks.sum(dim=-1) == 0.0):
        raise ValueError("Strategy action masks must contain at least one legal action per row.")
    if torch.any(detached_targets < 0.0) or torch.any(detached_targets[masks == 0.0] != 0.0):
        raise ValueError("Strategy targets must be non-negative and zero on illegal actions.")

    legal_targets = detached_targets * masks
    legal_target_mass = legal_targets.sum(dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(legal_target_mass)) or torch.any(legal_target_mass <= 0.0):
        raise ValueError("Strategy targets must assign positive total mass to legal actions.")
    if not torch.allclose(legal_target_mass, torch.ones_like(legal_target_mass), atol=1e-5, rtol=1e-5):
        raise ValueError("Strategy targets must be normalized over legal actions.")
    legal_targets = legal_targets / legal_target_mass

    masked_logits = logits.masked_fill(masks == 0.0, torch.finfo(logits.dtype).min)
    legal_log_probabilities = torch.log_softmax(masked_logits, dim=-1)
    per_sample_loss = -(legal_targets * legal_log_probabilities).sum(dim=-1)
    if sample_weights is None:
        weights = torch.ones_like(per_sample_loss)
    else:
        weights = sample_weights.detach().to(device=logits.device, dtype=logits.dtype)
        if weights.shape != per_sample_loss.shape:
            raise ValueError(
                "Strategy sample weights must have one scalar per batch row; "
                f"expected {tuple(per_sample_loss.shape)}, got {tuple(weights.shape)}."
            )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("Strategy sample weights must be finite and positive.")
    loss = (weights * per_sample_loss).sum() / weights.sum()
    if not torch.isfinite(loss):
        raise RuntimeError("Masked strategy cross-entropy became non-finite.")
    return loss


def train_strategy_network(
    network: StrategyNetwork,
    optimizer: torch.optim.Optimizer,
    batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]
) -> float:
    """Train on legal-only strategy distributions with iteration-weighted CE."""
    if len(batch) == 0:
        return 0.0

    network.train()

    if any(not isinstance(item, tuple) or len(item) != 4 for item in batch):
        raise ValueError(
            "Strategy training samples must be "
            "(observation, target, action_mask, iteration_weight) tuples."
        )
    states_list = [np.asarray(item[0], dtype=np.float32) for item in batch]
    targets_list = [np.asarray(item[1], dtype=np.float32) for item in batch]
    masks_list = [np.asarray(item[2], dtype=np.float32) for item in batch]
    weights_list = [float(item[3]) for item in batch]
    if any(state.shape != (OBSERVATION_DIM,) for state in states_list):
        raise ValueError(f"Strategy samples must use Phase 2 states shaped ({OBSERVATION_DIM},).")
    if any(target.shape != (ACTION_DIM,) for target in targets_list):
        raise ValueError(f"Strategy targets must use the Phase 2 {ACTION_DIM}-action shape.")
    if any(mask.shape != (ACTION_DIM,) for mask in masks_list):
        raise ValueError(f"Strategy masks must use the Phase 2 {ACTION_DIM}-action shape.")
    if not np.all(np.isfinite(weights_list)) or np.any(np.asarray(weights_list) <= 0.0):
        raise ValueError("Strategy iteration weights must be finite and positive.")

    states_t = torch.tensor(np.array(states_list), dtype=torch.float32, device=network.device)
    targets_t = torch.tensor(np.array(targets_list), dtype=torch.float32, device=network.device)
    masks_t = torch.tensor(np.array(masks_list), dtype=torch.float32, device=network.device)
    weights_t = torch.tensor(np.array(weights_list), dtype=torch.float32, device=network.device)

    logits = network(states_t)
    loss = masked_strategy_cross_entropy_loss(logits, targets_t, masks_t, weights_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
