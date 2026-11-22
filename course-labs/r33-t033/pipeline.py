"""
Course Lab R33-T033: Reinforcement Learning Three-State Chain & Minimal Baseline.

Student: Snehakumari Vinod Singh
Task: R33-T033 (Task 3 of 10)
Target Item: 18_reinforcement_learning.ipynb (Neural Network Policies)

Disclaimer:
The three-state chain environment and synthetic transition fixtures are designed
strictly for CPU smoke testing and baseline comparison. They are NOT real-world
reinforcement learning benchmarks and must not be used as evidence for real-world
or clinical predictive performance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import csv
import json
import random
from pathlib import Path


@dataclass
class ValidationError:
    """Represents a specific schema validation failure for a transition record."""
    episode_id: Any
    field: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "field": self.field,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return f"ValidationError(episode_id={self.episode_id!r}, field={self.field!r}, reason={self.reason!r})"


class ThreeStateChainEnv:
    """
    A three-state Markov chain environment (states: 0, 1, 2):
      - Initial state: 0
      - Terminal state: 2 (yields entry reward 1.0)
      - Action 'right': advances state by +1, clamped at state 2.
      - Action 'left': retreats state by -1, floor of state 0.
      - Step cap: 5 actions (yields reward 0.0 upon timeout).
    """

    def __init__(self, step_cap: int = 5) -> None:
        self.step_cap = step_cap
        self.state = 0
        self.steps = 0
        self.done = False

    def reset(self) -> int:
        self.state = 0
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action: str) -> Tuple[int, float, bool, Dict[str, Any]]:
        if self.done:
            return self.state, 0.0, True, {"reason": "Episode already terminated."}

        action_norm = action.strip().lower() if isinstance(action, str) else action
        if action_norm not in ("right", "left"):
            raise ValueError(f"Invalid action '{action}'. Allowed actions: ('right', 'left').")

        self.steps += 1
        reward = 0.0

        if action_norm == "right":
            self.state = min(2, self.state + 1)
        elif action_norm == "left":
            self.state = max(0, self.state - 1)

        if self.state == 2:
            reward = 1.0
            self.done = True
        elif self.steps >= self.step_cap:
            reward = 0.0
            self.done = True

        return self.state, reward, self.done, {"steps": self.steps}

    def run_trajectory(self, actions: List[str]) -> Dict[str, Any]:
        """Execute a predefined action sequence from the reset state."""
        self.reset()
        total_reward = 0.0
        trajectory = [self.state]
        transitions: List[Dict[str, Any]] = []

        for a in actions:
            curr_state = self.state
            next_state, r, done, _ = self.step(a)
            total_reward += r
            trajectory.append(next_state)
            transitions.append({
                "state": curr_state,
                "action": a,
                "next_state": next_state,
                "reward": r,
                "terminated": done,
            })
            if done:
                break

        return {
            "final_state": self.state,
            "total_reward": total_reward,
            "steps": self.steps,
            "terminated": self.done,
            "trajectory": trajectory,
            "transitions": transitions,
        }


class AlwaysRightPolicy:
    """
    Minimal honest baseline policy for the three-state chain contract.
    Always selects the 'right' action to advance toward the terminal state.
    """

    def __init__(self) -> None:
        self.name = "AlwaysRightPolicy"

    def predict(self, state: int) -> str:
        return "right"


class AlwaysLeftPolicy:
    """Comparison baseline policy that always selects 'left'."""

    def __init__(self) -> None:
        self.name = "AlwaysLeftPolicy"

    def predict(self, state: int) -> str:
        return "left"


class RandomPolicy:
    """Comparison baseline policy that randomly chooses actions."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.name = "RandomPolicy"
        self.rng = random.Random(seed)

    def predict(self, state: int) -> str:
        return self.rng.choice(["left", "right"])


def evaluate_policy(
    policy: Any,
    env: ThreeStateChainEnv,
    n_episodes: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a policy on the environment over multiple episodes.
    Returns metrics: mean_reward, mean_steps, success_rate.
    """
    total_rewards: List[float] = []
    episode_lengths: List[int] = []
    successes = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state

        total_rewards.append(ep_reward)
        episode_lengths.append(env.steps)
        if state == 2 and ep_reward == 1.0:
            successes += 1

    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "episodes": n_episodes,
        "mean_reward": sum(total_rewards) / n_episodes,
        "mean_steps": sum(episode_lengths) / n_episodes,
        "success_rate": successes / n_episodes,
    }


class SchemaValidator:
    """
    Validates RL transition records against the contract:
      - episode_id: string or int, non-empty.
      - state: integer in {0, 1, 2}.
      - action: string in {'right', 'left'}.
      - next_state: integer in {0, 1, 2}.
      - reward: float/int, 1.0 on terminal entry, 0.0 otherwise.
      - terminated: bool.
    """

    ALLOWED_ACTIONS = {"right", "left"}
    VALID_STATES = {0, 1, 2}

    def validate_transition(self, record: Any) -> List[ValidationError]:
        if not isinstance(record, dict):
            return [ValidationError(
                episode_id=None,
                field="record",
                reason=f"Expected transition record dict, got {type(record).__name__}.",
            )]

        ep_id = record.get("episode_id")
        errors: List[ValidationError] = []

        # 1. episode_id
        if ep_id is None or (isinstance(ep_id, str) and not ep_id.strip()):
            errors.append(ValidationError(
                episode_id=ep_id,
                field="episode_id",
                reason="Missing or empty episode_id.",
            ))

        # 2. state
        state = record.get("state")
        if state not in self.VALID_STATES:
            errors.append(ValidationError(
                episode_id=ep_id,
                field="state",
                reason=f"Invalid state: {state!r} not in {sorted(self.VALID_STATES)}.",
            ))

        # 3. action
        action = record.get("action")
        if not isinstance(action, str) or action.lower() not in self.ALLOWED_ACTIONS:
            errors.append(ValidationError(
                episode_id=ep_id,
                field="action",
                reason=f"Invalid action: {action!r} not in {sorted(self.ALLOWED_ACTIONS)}.",
            ))

        # 4. next_state
        next_state = record.get("next_state")
        if next_state not in self.VALID_STATES:
            errors.append(ValidationError(
                episode_id=ep_id,
                field="next_state",
                reason=f"Invalid next_state: {next_state!r} not in {sorted(self.VALID_STATES)}.",
            ))

        # 5. reward
        reward = record.get("reward")
        if reward is None or isinstance(reward, bool) or not isinstance(reward, (int, float)):
            errors.append(ValidationError(
                episode_id=ep_id,
                field="reward",
                reason=f"Invalid reward type: {reward!r}.",
            ))
        elif reward not in (0.0, 1.0):
            errors.append(ValidationError(
                episode_id=ep_id,
                field="reward",
                reason=f"Reward out of bounds: expected 0.0 or 1.0, got {reward}.",
            ))

        # 6. terminated
        terminated = record.get("terminated")
        if not isinstance(terminated, bool):
            errors.append(ValidationError(
                episode_id=ep_id,
                field="terminated",
                reason=f"Invalid terminated flag: expected bool, got {type(terminated).__name__}.",
            ))

        # 7. Environment physics consistency check
        if state in self.VALID_STATES and isinstance(action, str) and action.lower() in self.ALLOWED_ACTIONS and next_state in self.VALID_STATES:
            act_norm = action.lower()
            expected_next = min(2, state + 1) if act_norm == "right" else max(0, state - 1)
            if next_state != expected_next:
                errors.append(ValidationError(
                    episode_id=ep_id,
                    field="next_state",
                    reason=f"Inconsistent transition: state={state} + action='{action}' should lead to next_state={expected_next}, got {next_state}.",
                ))
            if next_state == 2 and reward != 1.0:
                errors.append(ValidationError(
                    episode_id=ep_id,
                    field="reward",
                    reason=f"Inconsistent reward: state 2 terminal transition must yield reward 1.0, got {reward}.",
                ))

        return errors

    def validate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[ValidationError]]:
        valid_recs: List[Dict[str, Any]] = []
        all_errors: List[ValidationError] = []

        for r in records:
            errs = self.validate_transition(r)
            if errs:
                all_errors.extend(errs)
            else:
                valid_recs.append(r)

        return valid_recs, all_errors


def load_dataset(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON must be a list of transition objects.")
            return data

    elif path.suffix.lower() == ".csv":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "episode_id": row["episode_id"],
                    "state": int(row["state"]),
                    "action": row["action"],
                    "next_state": int(row["next_state"]),
                    "reward": float(row["reward"]),
                    "terminated": row["terminated"].strip().lower() in ("true", "1"),
                })
        return records

    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
