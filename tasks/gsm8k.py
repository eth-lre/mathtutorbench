from typing import Dict, List, Union
import re
from registry import TaskRegistry
from .base import Task

filter_list = [
            {
                "name": "strict-match",
                "filter": [
                    {
                        "function": "regex",
                        "regex_pattern": r"#### (\\-?[0-9\\.\\,]+)"
                    }
                ]
            },
            {
                "name": "flexible-extract",
                "filter": [
                    {
                        "function": "regex",
                        "group_select": -1,
                        "regex_pattern": r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
                    }
                ]
            }
        ]

@TaskRegistry.register("problem_solving")
class GSM8K(Task):

    def parse_response(self, response: str) -> float:
        """Extract the final answer from the model's response"""
        numbers = re.findall(r'-?\d*\.?\d+', response)
        if numbers:
            return str(numbers[-1])
        return None

    def apply_regex(self, text: str, filter_config: Dict) -> List[str]:
        """Applies the regex defined in the filter config to the text."""
        pattern = filter_config.get("regex_pattern")
        group_select = filter_config.get("group_select", 0)

        matches = re.findall(pattern, text)

        # Flatten matches and select the appropriate group if applicable
        if matches:
            if isinstance(matches[0], tuple):
                matches = [match[group_select] if group_select >= 0 else match[-1] for match in matches]
            else:
                matches = matches

        return matches

    def process_predictions(self, predictions: List[str], filter_name: str) -> List[float]:
        """Process predictions based on the specified filter."""
        filter_config = next(
            (f["filter"][0] for f in filter_list if f["name"] == filter_name),
            None
        )

        if not filter_config:
            raise ValueError(f"Filter '{filter_name}' not found in filter list.")

        processed_predictions = []
        for prediction in predictions:
            if prediction is None:
                processed_predictions.append(None)
                continue

            matches = self.apply_regex(prediction, filter_config)
            if matches:
                # Convert the first matched value to float (remove commas if present)
                try:
                    processed_predictions.append(float(matches[0].replace(",", "")))
                except ValueError:
                    processed_predictions.append(None)
            else:
                processed_predictions.append(None)

        return processed_predictions

    def compute_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Compute accuracy and other metrics for GSM8K task."""
        results = {}

        for filter_name in ["strict-match", "flexible-extract"]:
            # processed_predictions = self.process_predictions(predictions, filter_name)

            # Change predictions to floats
            processed_predictions = [float(p) if p is not None else None for p in predictions]

            # Change targets to floats
            numeric_targets = [float(t) for t in targets]

            print(processed_predictions)
            print(numeric_targets)

            correct = sum(
                1 for p, t in zip(processed_predictions, numeric_targets)
                if p is not None and abs(p - t) < 1e-6
            )
            total = len(numeric_targets)
            accuracy = correct / total if total > 0 else 0.0

            results[f"accuracy_{filter_name}"] = accuracy

        return results