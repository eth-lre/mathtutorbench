from typing import Dict, List, Any
import re
from registry import TaskRegistry
from dataloaders.verifiers import StepVerifyDataset
from .base import Task, TaskConfig


@TaskRegistry.register("mistake_correction")
class MistakeCorrectionTask(Task):
    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_dataset(self) -> None:
        """Load and preprocess the verifiers dataset"""
        self.train_dataset = StepVerifyDataset(self.config.dataset_path).load()
        self.test_dataset = StepVerifyDataset(self.config.dataset_path).load()

    def parse_response(self, response: str) -> float:
        """Extract the final answer from the model's response"""
        # First try to find "Final Answer: X" format with optional $ and commas
        final_answer_match = re.search(r'Final Answer:\s*\$?([-,\d]*\.?\d+)', response)
        if final_answer_match:
            try:
                return float(final_answer_match.group(1).replace(",", ""))
            except ValueError:
                return None

        # If no explicit final answer, find the last number with optional $ and commas
        numbers = re.findall(r'\$?([-,\d]*\.?\d+)', response)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                return None

        return None

    def compute_metrics(self, predictions: List[float], targets: List[str]) -> Dict[str, float]:
        """Compute accuracy metrics"""
        # Convert predictions and targets to floats
        processed_predictions = [float(p) if p is not None else None for p in predictions]
        numeric_targets = [float(t) for t in targets]

        # Count correct predictions (within small epsilon for floating point comparison)
        correct = sum(
            1 for p, t in zip(processed_predictions, numeric_targets)
            if p is not None and abs(p - t) < 1e-6
        )
        total = len(numeric_targets)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy
        }