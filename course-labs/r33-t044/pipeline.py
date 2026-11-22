"""
Course Lab R33-T044: Evaluation Metrics & Three-State RL Chain Simulation
Implements reproducible RL chain dynamics and explicit metric evaluations.
"""
from typing import List, Dict, Any, Tuple

class ThreeStateChainEnv:
    """
    A three-state chain (0, 1, 2):
    - Starts at state 0.
    - 'right' advances state by 1.
    - 'left' retreats with floor 0.
    - State 2 is terminal with reward 1.
    - Step cap of 5 actions.
    """
    def __init__(self, step_cap: int = 5):
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
            return self.state, 0.0, True, {"reason": "Already terminal"}

        self.steps += 1
        reward = 0.0

        if action == "right":
            self.state = min(2, self.state + 1)
        elif action == "left":
            self.state = max(0, self.state - 1)
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.state == 2:
            reward = 1.0
            self.done = True
        elif self.steps >= self.step_cap:
            self.done = True

        return self.state, reward, self.done, {"steps": self.steps}

    def run_trajectory(self, actions: List[str]) -> Dict[str, Any]:
        self.reset()
        total_reward = 0.0
        trajectory = [self.state]
        for a in actions:
            state, r, done, _ = self.step(a)
            total_reward += r
            trajectory.append(state)
            if done:
                break
        return {
            "final_state": self.state,
            "total_reward": total_reward,
            "steps": self.steps,
            "done": self.done,
            "trajectory": trajectory
        }

def calculate_precision_recall(y_true: List[int], y_pred: List[int], zero_division: float = 0.0) -> Tuple[float, float]:
    """
    Calculates precision and recall with explicit zero-denominator safety.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    return precision, recall

def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculates Mean Absolute Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths must match")
    if not y_true:
        raise ValueError("Empty sequence")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
