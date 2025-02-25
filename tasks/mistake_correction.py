from typing import Dict, List, Any
import re

from dataloaders.base import HuggingFaceDataset
from registry import TaskRegistry
from .base import Task, TaskConfig


@TaskRegistry.register("mistake_correction")
class MistakeCorrectionTask(Task):
    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_dataset(self) -> None:
        """Load and preprocess the verifiers dataset"""
        self.train_dataset = HuggingFaceDataset(self.config.dataset_path, self.config.dataset_name,
                                                split=self.config.training_split).load()
        self.test_dataset = HuggingFaceDataset(self.config.dataset_path, self.config.dataset_name,
                                               split=self.config.test_split).load()
        self.train_dataset = self._format_dataset(self.train_dataset)
        self.test_dataset = self._format_dataset(self.test_dataset)

    def _format_dataset(self, raw_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        processed_examples = []
        for example in raw_examples:
            # Process dialog history
            conversation_list = example.get("dialog_history", [])
            formatted_dialog = []
            cutoff = len(conversation_list)

            # Find the cutoff point where the student response starts
            for i, turn in enumerate(conversation_list):
                if turn["user"] == "Student":
                    cutoff = i
                    break
                formatted_dialog.append(f"Teacher: {turn['text']}")

            dialog_history_str = "\n".join(formatted_dialog)

            error_example = {
                'question': example['problem'],
                'student_solution': "\\n".join(["Step " + str(sub_index + 1) + " - " + substep for sub_index, substep in
                                                enumerate(example["student_incorrect_solution"][:-1])]),
                'is_error': True,
                'error_step': int(example['incorrect_index']) + 1,  # Convert to 1-based indexing
                'dialog_history': dialog_history_str,
                "student_chat_solution": conversation_list[cutoff]['text'],
                "reference_solution": example["reference_solution"],
            }
            processed_examples.append(error_example)

        return processed_examples

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