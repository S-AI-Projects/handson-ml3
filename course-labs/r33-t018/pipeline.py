import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def validate_fixture(data):
    if data.get("dataset_type") != "synthetic_simulated":
        raise ValueError("Dataset must be explicitly labelled synthetic_simulated")

    if data.get("gamma") != 1.0:
        raise ValueError("This fixture expects gamma = 1.0")

    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Episodes must be a non-empty list")

    seen_ids = set()
    seen_splits = set()

    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("Each episode must be an object")

        episode_id = episode.get("episode_id")
        split = episode.get("split")
        transitions = episode.get("transitions")

        if not episode_id:
            raise ValueError("Missing episode_id")

        if split not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")

        seen_splits.add(split)

        if not isinstance(transitions, list) or not transitions:
            raise ValueError("Each episode needs transitions")

        for transition in transitions:
            required = {
                "record_id", "state", "action", "next_state",
                "reward", "terminated", "predicted_value"
            }

            if not required.issubset(transition):
                raise ValueError("Transition is missing required fields")

            record_id = transition["record_id"]

            if not isinstance(record_id, str) or not record_id:
                raise ValueError("record_id must be a non-empty string")

            if record_id in seen_ids:
                raise ValueError("Duplicate record_id: " + record_id)

            seen_ids.add(record_id)

            if transition["action"] not in {"left", "right"}:
                raise ValueError("Invalid action")

            if not isinstance(transition["reward"], (int, float)):
                raise ValueError("Reward must be numeric")

            if not isinstance(transition["terminated"], bool):
                raise ValueError("terminated must be boolean")

            if not isinstance(transition["predicted_value"], (int, float)):
                raise ValueError("predicted_value must be numeric")

    if seen_splits != {"train", "validation", "test"}:
        raise ValueError("Train, validation and test splits are required")


def analyse(data):
    validate_fixture(data)

    report = []
    total = 0

    for episode in data["episodes"]:
        transitions = episode["transitions"]
        expected_values = [0.0] * len(transitions)

        running_return = 0.0

        for index in range(len(transitions) - 1, -1, -1):
            transition = transitions[index]

            if transition["terminated"]:
                running_return = transition["reward"]
            else:
                running_return += transition["reward"]

            expected_values[index] = running_return

        for transition, expected in zip(transitions, expected_values):
            total += 1
            predicted = transition["predicted_value"]

            if abs(predicted - expected) > 1e-9:
                report.append({
                    "record_id": transition["record_id"],
                    "episode_id": episode["episode_id"],
                    "split": episode["split"],
                    "expected_value": expected,
                    "predicted_value": predicted,
                    "reason_category": (
                        "overestimated_zero_return"
                        if expected == 0 and predicted > 0
                        else "prediction_mismatch"
                    )
                })

    return {
        "dataset_type": data["dataset_type"],
        "total_records": total,
        "error_count": len(report),
        "correct_count": total - len(report),
        "errors": report
    }


def main():
    with FIXTURE_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    result = analyse(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
