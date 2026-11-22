"""Instrumented, bounded External-Sampling MCCFR with policy rollouts.

The explicit tree remains External-Sampling MCCFR: every legal traverser action
is enumerated and every opponent action is sampled from regret matching.  At
``max_steps``, a trajectory is sampled under the same current two-player policy
profile until a genuine terminal payoff is reached.  Its conditional expected
payoff is the continuation value for the retained tree node, so it does not
introduce a fabricated leaf utility or bias that node's Monte-Carlo target.

Because information sets deeper than ``max_steps`` are not explicitly visited
or updated, this is depth-limited/approximate External-Sampling MCCFR, not
exact full-depth MCCFR.  The approximation is the finite update horizon and
the resulting rollout variance, not a switch to Random Blue or random
traverser-action selection in the explicit MCCFR tree.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from pokemon_rl.agents.deep_cfr import (
    _game_action_to_index,
    create_game_state_snapshot,
    get_information_set_observation,
    get_legal_action_mask,
    regret_matching,
)
from pokemon_rl.cards.energy import EnergyCard
from pokemon_rl.cards.pokemon import PokemonCard
from pokemon_rl.cards.trainer import TrainerCard
from pokemon_rl.environment import ACTION_DIM, OBSERVATION_DIM
from pokemon_rl.game import Action, GameEngine, GameState


ProgressCallback = Callable[[Dict[str, Any]], None]
RolloutTraceCallback = Callable[[Dict[str, Any]], None]

_ACTION_SPACE_SIZE = ACTION_DIM
_REPRESENTABLE_HAND_SLOTS = 8


class CFRDiagnostics:
    """Lightweight counters for traversal correctness and performance."""

    def __init__(self, progress_every_nodes: int = 0, progress_callback: Optional[ProgressCallback] = None):
        self.decision_nodes = 0
        self.traverser_decision_nodes = 0
        self.opponent_decision_nodes = 0
        self.terminal_nodes = 0
        self.chance_samples = 0
        self.max_depth = 0
        self.regret_samples = 0
        self.strategy_samples = 0
        self.recursive_calls = 0
        self.state_copies = 0
        self.game_transitions = 0
        self.legal_action_total = 0
        self.legal_action_observations = 0
        self.horizon_rollouts = 0
        self.rollout_steps = 0
        self.progress_every_nodes = progress_every_nodes
        self.progress_callback = progress_callback

    def note_decision(self, player: int, traverser: int, legal_count: int, depth: int) -> None:
        self.decision_nodes += 1
        if player == traverser:
            self.traverser_decision_nodes += 1
        else:
            self.opponent_decision_nodes += 1
        self.legal_action_total += legal_count
        self.legal_action_observations += 1
        if self.progress_callback and self.progress_every_nodes and self.decision_nodes % self.progress_every_nodes == 0:
            self.progress_callback(self.snapshot(depth))

    def snapshot(self, depth: int) -> Dict[str, Any]:
        return {
            "decision_nodes": self.decision_nodes,
            "recursive_calls": self.recursive_calls,
            "max_depth": max(depth, self.max_depth),
            "state_copies": self.state_copies,
            "game_transitions": self.game_transitions,
            "horizon_rollouts": self.horizon_rollouts,
        }


def action_to_index(action: Action) -> int:
    """Map a concrete GameEngine action to its approved Phase 2 0--69 index."""
    return _game_action_to_index(action)


def legal_actions_and_mask(state: GameState) -> Tuple[np.ndarray, np.ndarray]:
    """Return the canonical, current-state Deep CFR action mask.

    The shared encoder derives its 70 entries directly from ``GameEngine``.
    It excludes hand slots beyond seven before assigning action groups,
    preserving slot semantics when a player holds nine or more cards.
    """
    mask = get_legal_action_mask(state, state.current_player_idx).astype(np.float32, copy=False)
    indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    return indices, mask


def _decode_action(action_idx: int) -> Action:
    """Decode one Phase 2 action index without claiming it is currently legal."""
    if action_idx == 0:
        return Action("end_turn")
    if action_idx == 1:
        return Action("attack", attack_idx=0)
    if action_idx == 2:
        return Action("attack", attack_idx=1)
    if action_idx == 3:
        return Action("retreat")
    if 4 <= action_idx <= 11:
        return Action("play_pokemon", card_idx=action_idx - 4, bench_slot=0)
    if 12 <= action_idx <= 19:
        return Action("play_pokemon", card_idx=action_idx - 12, bench_slot=1)
    if 20 <= action_idx <= 27:
        return Action("attach_energy", card_idx=action_idx - 20, target="active")
    if 28 <= action_idx <= 35:
        return Action("attach_energy", card_idx=action_idx - 28, bench_slot=0, target="bench")
    if 36 <= action_idx <= 43:
        return Action("attach_energy", card_idx=action_idx - 36, bench_slot=1, target="bench")
    if 44 <= action_idx <= 51:
        return Action("play_potion", card_idx=action_idx - 44, target="active")
    if 52 <= action_idx <= 59:
        return Action("play_potion", card_idx=action_idx - 52, bench_slot=0, target="bench")
    if 60 <= action_idx <= 67:
        return Action("play_potion", card_idx=action_idx - 60, bench_slot=1, target="bench")
    if action_idx == 68:
        return Action("promote", bench_slot=0)
    if action_idx == 69:
        return Action("promote", bench_slot=1)
    raise ValueError(f"Action index {action_idx} is out of range [0, {ACTION_DIM - 1}].")


def _action_description(action_idx: int) -> Dict[str, Any]:
    """Return the action-space meaning used by optional continuation traces."""
    action = _decode_action(action_idx)
    return {
        "action_type": action.action_type,
        "hand_slot": action.card_idx,
        "bench_slot": action.bench_slot,
        "attack_idx": action.attack_idx,
        "target": action.target,
    }


def _card_description(card: Any) -> Optional[Dict[str, Any]]:
    """Keep trace output simple and non-mutating while identifying a card."""
    if card is None:
        return None
    return {
        "class_name": type(card).__name__,
        "name": getattr(card, "name", None),
        "card_type": getattr(card, "card_type", None),
        "effect_type": getattr(card, "effect_type", None),
    }


def _validate_action_against_current_state(state: GameState, action_idx: int) -> None:
    """Reject an action that is not legal for the state immediately pre-step.

    This is deliberately a validation, not recovery logic: it recomputes the
    current player's observation-independent legal mask and verifies the card
    type selected by an indexed hand action.  A mismatch exposes an invariant
    violation at the call site instead of letting a stale/mis-mapped index be
    interpreted as a different GameEngine action.
    """
    if not 0 <= int(action_idx) < _ACTION_SPACE_SIZE:
        raise ValueError(f"Action index {action_idx} is out of range [0, {ACTION_DIM - 1}].")

    current_player = state.current_player_idx
    current_mask = get_legal_action_mask(state, current_player)
    if current_mask[int(action_idx)] != 1.0:
        raise RuntimeError(
            f"Action {action_idx} is not legal for current player {current_player}; "
            "the action was selected from a stale or invalid legal mask."
        )

    action = _decode_action(int(action_idx))
    # The direct GameEngine equality check validates target/Bench/attack fields
    # as well as the index. It prevents a stale slot from becoming another
    # representable action after a hand mutation.
    if action not in GameEngine.get_legal_actions(state):
        raise RuntimeError(
            f"Action {action_idx} decoded to {action!r}, which is not a current legal GameEngine action."
        )
    if action.card_idx is None:
        return

    hand = state.players[current_player].hand
    slot = action.card_idx
    if not 0 <= slot < len(hand):
        raise RuntimeError(
            f"Legal action {action_idx} references missing hand slot {slot} for player {current_player}."
        )
    card = hand[slot]
    if action.action_type == "play_pokemon" and not isinstance(card, PokemonCard):
        raise RuntimeError(
            f"Action {action_idx} means play_pokemon at slot {slot}, but that slot holds {card!r}."
        )
    if action.action_type == "attach_energy" and not isinstance(card, EnergyCard):
        raise RuntimeError(
            f"Action {action_idx} means attach_energy at slot {slot}, but that slot holds {card!r}."
        )
    if (action.action_type == "play_potion"
            and (not isinstance(card, TrainerCard) or card.effect_type != "heal")):
        raise RuntimeError(
            f"Action {action_idx} means play_potion at slot {slot}, but that slot holds {card!r}."
        )


def apply_action(state: GameState, action_idx: int, diagnostics: Optional[CFRDiagnostics] = None) -> GameState:
    """Validate and apply one indexed action to the caller-owned state."""
    _validate_action_against_current_state(state, action_idx)
    action = _decode_action(action_idx)
    if diagnostics is not None:
        diagnostics.game_transitions += 1
    return GameEngine.step(state, action)


def get_player_regrets(regret_net: Any, obs: np.ndarray, player_idx: int) -> np.ndarray:
    """Query the requested player's current regret approximation."""
    network = regret_net[player_idx] if isinstance(regret_net, (list, tuple, dict)) else regret_net
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Deep CFR information state must have shape ({OBSERVATION_DIM},), got {obs.shape}.")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=network.device).unsqueeze(0)
    with torch.no_grad():
        regrets = network(obs_tensor).squeeze(0).cpu().numpy()
    if regrets.shape != (ACTION_DIM,):
        raise RuntimeError(
            f"Regret network returned shape {regrets.shape}; expected ({ACTION_DIM},) for Phase 2."
        )
    return regrets


def _policy_for_current_player(state: GameState, regret_net: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the active player's observation, legal mask, and policy."""
    indices, mask = legal_actions_and_mask(state)
    if indices.size == 0:
        raise RuntimeError("A non-terminal GameState has no legal Phase 2 action.")
    player = state.current_player_idx
    observation = get_information_set_observation(state, player)
    return indices, mask, observation, regret_matching(get_player_regrets(regret_net, observation, player), mask)


def _sample_legal_action(indices: np.ndarray, strategy: np.ndarray, context: str) -> int:
    """Sample only the currently legal support of a 70-action strategy vector."""
    indices = np.asarray(indices, dtype=np.int64)
    strategy = np.asarray(strategy, dtype=np.float64)
    if strategy.shape != (ACTION_DIM,):
        raise ValueError(f"{context} strategy must have shape ({ACTION_DIM},), got {strategy.shape}.")
    if indices.size == 0:
        raise RuntimeError(f"{context} cannot sample because the state has no legal actions.")
    legal_probabilities = strategy[indices]
    if np.any(legal_probabilities < 0.0) or not np.all(np.isfinite(legal_probabilities)):
        raise RuntimeError(f"{context} received invalid legal-action probabilities.")
    probability_sum = float(legal_probabilities.sum())
    if not np.isclose(probability_sum, 1.0, atol=1e-6):
        raise RuntimeError(f"{context} legal-action probabilities must sum to one, got {probability_sum}.")
    if np.any(strategy[np.setdiff1d(np.arange(ACTION_DIM), indices)] != 0.0):
        raise RuntimeError(f"{context} strategy assigned probability to an illegal action.")
    return int(np.random.choice(indices, p=legal_probabilities / probability_sum))


def _terminal_utility(state: GameState, traverser: int) -> float:
    """Return the exact zero-sum terminal payoff from the traverser's view."""
    if not state.game_over or state.winner_idx not in (0, 1):
        raise RuntimeError("Terminal utility requested before GameState recorded a winner.")
    return 1.0 if state.winner_idx == traverser else -1.0


def rollout_to_terminal(state: GameState, traverser: int, regret_net: Any, diagnostics: CFRDiagnostics,
                        trace_callback: Optional[RolloutTraceCallback] = None) -> float:
    """Sample a real terminal continuation from the current policy profile.

    Action selection uses each player's current regret network, that player's
    information-set observation, and its legal-action mask.  Conditional on
    this GameState, the expected terminal payoff equals its continuation value
    under that profile.  Thus it is unbiased for a retained pre-horizon node's
    value, though it adds variance.  The boundary still makes the overall
    method depth-limited/approximate because no regret samples are stored for
    information sets visited only during this rollout.
    """
    diagnostics.horizon_rollouts += 1
    while not state.game_over:
        # These are intentionally recomputed after every transition.  A draw,
        # attach, potion, attack, or turn change can alter both the active
        # player and which fixed hand-slot actions are legal.
        indices, mask, observation, strategy = _policy_for_current_player(state, regret_net)
        action_index = _sample_legal_action(indices, strategy, "Continuation rollout")
        if trace_callback is not None:
            player = state.current_player_idx
            network = regret_net[player] if isinstance(regret_net, (list, tuple, dict)) else regret_net
            action_info = _action_description(action_index)
            slot = action_info["hand_slot"]
            selected_card = state.players[player].hand[slot] if slot is not None else None
            trace_callback({
                "current_player": player,
                "observation": observation.copy(),
                "legal_action_indices": indices.copy(),
                "legal_mask": mask.copy(),
                "regret_network_player": player,
                "regret_network": {
                    "type": type(network).__name__,
                    "device": str(network.device),
                },
                "selected_action_index": action_index,
                "selected_action": action_info,
                "selected_card": _card_description(selected_card),
                "hand_before": [_card_description(card) for card in state.players[player].hand],
            })
        apply_action(state, action_index, diagnostics)
        diagnostics.rollout_steps += 1
    diagnostics.terminal_nodes += 1
    return _terminal_utility(state, traverser)


def _store_samples(traverser: int, observation: np.ndarray, regret_target: np.ndarray, strategy: np.ndarray,
                   action_mask: np.ndarray, iteration: int,
                   depth: int, regret_buffer: Any, strategy_buffer: Any,
                   diagnostics: CFRDiagnostics, strategy_coverage_audit: Optional[Any] = None) -> None:
    if observation.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Deep CFR samples require ({OBSERVATION_DIM},)-dimensional information states.")
    if regret_target.shape != (ACTION_DIM,) or strategy.shape != (ACTION_DIM,) or action_mask.shape != (ACTION_DIM,):
        raise ValueError(f"Deep CFR targets require the Phase 2 ({ACTION_DIM},) action shape.")
    if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(regret_target)) or not np.all(np.isfinite(strategy)):
        raise ValueError("Deep CFR cannot store non-finite samples.")
    if not np.all((action_mask == 0.0) | (action_mask == 1.0)) or np.sum(action_mask) == 0.0:
        raise ValueError("Deep CFR strategy samples require a non-empty binary legal-action mask.")
    if not np.isclose(float(strategy.sum()), 1.0):
        raise ValueError("Deep CFR strategy targets must sum to one.")
    if np.any(strategy[action_mask == 0.0] != 0.0) or np.any(regret_target[action_mask == 0.0] != 0.0):
        raise ValueError("Deep CFR cannot store targets for illegal actions.")
    if not isinstance(iteration, (int, np.integer)) or iteration <= 0:
        raise ValueError("Deep CFR strategy sample iteration weights must be positive integers.")
    if regret_buffer is not None:
        (regret_buffer[traverser] if isinstance(regret_buffer, (list, tuple, dict)) else regret_buffer).add((observation, regret_target))
        diagnostics.regret_samples += 1
    if strategy_buffer is not None:
        # Preserve the exact pre-action legality used to create ``strategy``.
        # A legal action can correctly have target probability zero, so the
        # mask cannot be reconstructed from target values during training. The
        # separate scalar records the outer iteration for weighted average-
        # strategy fitting; it never changes the normalized target itself.
        strategy_reservoir = (
            strategy_buffer[traverser]
            if isinstance(strategy_buffer, (list, tuple, dict))
            else strategy_buffer
        )
        retained_slot = strategy_reservoir.add(
            (observation, strategy, action_mask, float(iteration))
        )
        if strategy_coverage_audit is not None:
            strategy_coverage_audit.record_strategy_sample(
                player=traverser,
                depth=depth,
                observation=observation,
                target=strategy,
                action_mask=action_mask,
                retained_slot=retained_slot,
            )
        diagnostics.strategy_samples += 1


def traverse(state: GameState, traverser: int, iteration: int, regret_net: Any, strategy_net: Any,
             regret_buffer: Any, strategy_buffer: Any, diagnostics: CFRDiagnostics,
             max_steps: int, depth: int = 0, strategy_coverage_audit: Optional[Any] = None) -> float:
    """Bounded External-Sampling MCCFR: enumerate traverser actions, sample opponents.

    ``depth >= max_steps`` delegates only the remaining continuation to
    :func:`rollout_to_terminal`; no explicit-tree traverser action is sampled.
    """
    diagnostics.recursive_calls += 1
    diagnostics.max_depth = max(diagnostics.max_depth, depth)
    if state.game_over:
        diagnostics.terminal_nodes += 1
        return _terminal_utility(state, traverser)
    if depth >= max_steps:
        return rollout_to_terminal(state, traverser, regret_net, diagnostics)

    player = state.current_player_idx
    observation = get_information_set_observation(state, player)
    indices, mask = legal_actions_and_mask(state)
    if indices.size == 0:
        raise RuntimeError("A non-terminal explicit-tree node has no legal Phase 2 action.")

    diagnostics.note_decision(player, traverser, int(indices.size), depth)
    strategy = regret_matching(get_player_regrets(regret_net, observation, player), mask)
    if player == traverser:
        action_utilities: Dict[int, float] = {}
        for action_index in indices:
            # Multiple sibling branches require isolation. This is the only
            # place deep copies are needed; opponent nodes have one child.
            diagnostics.state_copies += 1
            child = create_game_state_snapshot(state)
            apply_action(child, int(action_index), diagnostics)
            action_utilities[int(action_index)] = traverse(child, traverser, iteration, regret_net, strategy_net,
                                                          regret_buffer, strategy_buffer, diagnostics,
                                                          max_steps, depth + 1, strategy_coverage_audit)
        # A convex combination of terminal payoffs must be in [-1, 1]. Float32
        # strategy normalization can exceed the endpoint by about 1e-7, so clip
        # only that numerical noise before emitting a regret target/diagnostic.
        expected_utility = float(np.clip(
            sum(strategy[action] * action_utilities[int(action)] for action in indices),
            -1.0,
            1.0,
        ))
        regret_target = np.zeros(ACTION_DIM, dtype=np.float32)
        for action_index in indices:
            regret_target[int(action_index)] = action_utilities[int(action_index)] - expected_utility
        _store_samples(traverser, observation, regret_target, strategy, mask, iteration, depth,
                       regret_buffer, strategy_buffer, diagnostics, strategy_coverage_audit)
        return expected_utility

    # At an opponent node external sampling visits exactly one child. In-place
    # mutation is safe because this state has no later sibling reuse.
    action_index = _sample_legal_action(indices, strategy, "External-sampling MCCFR")
    apply_action(state, action_index, diagnostics)
    return traverse(state, traverser, iteration, regret_net, strategy_net, regret_buffer, strategy_buffer,
                    diagnostics, max_steps, depth + 1, strategy_coverage_audit)


def run_cfr_traversal(root_state: GameState, traverser: int, regret_net: Any, strategy_net: Any,
                      regret_buffer: Any, strategy_buffer: Any, iteration: int = 1,
                      max_steps: int = 4, progress_every_nodes: int = 0,
                      progress_callback: Optional[ProgressCallback] = None,
                      strategy_coverage_audit: Optional[Any] = None) -> Tuple[float, Dict[str, Any]]:
    """Run one profiled, horizon-bounded external-sampling MCCFR traversal."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    started = time.perf_counter()
    diagnostics = CFRDiagnostics(progress_every_nodes, progress_callback)
    diagnostics.state_copies += 1
    value = traverse(create_game_state_snapshot(root_state), traverser, iteration, regret_net, strategy_net,
                     regret_buffer, strategy_buffer, diagnostics, max_steps,
                     strategy_coverage_audit=strategy_coverage_audit)
    elapsed_seconds = time.perf_counter() - started
    average_legal_action_count = (diagnostics.legal_action_total / diagnostics.legal_action_observations
                                  if diagnostics.legal_action_observations else 0.0)
    return float(value), {
        "decision_nodes": diagnostics.decision_nodes,
        "traverser_decision_nodes": diagnostics.traverser_decision_nodes,
        "opponent_decision_nodes": diagnostics.opponent_decision_nodes,
        "terminal_nodes": diagnostics.terminal_nodes,
        "chance_samples": diagnostics.chance_samples,
        "max_depth": diagnostics.max_depth,
        "regret_samples": diagnostics.regret_samples,
        "strategy_samples": diagnostics.strategy_samples,
        "recursive_calls": diagnostics.recursive_calls,
        "state_copies": diagnostics.state_copies,
        "game_transitions": diagnostics.game_transitions,
        "average_legal_action_count": average_legal_action_count,
        "horizon_rollouts": diagnostics.horizon_rollouts,
        "rollout_steps": diagnostics.rollout_steps,
        "elapsed_seconds": elapsed_seconds,
        "traverser": traverser,
        "root_utility": float(value),
        "max_steps": max_steps,
    }
