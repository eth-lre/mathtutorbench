from typing import Dict, List
from registry import TaskRegistry
from dataloaders.verifiers import StepVerifyDataset
from .base import Task, TaskConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@TaskRegistry.register("solution_correctness")
class SolutionCorrectnessTask(Task):
    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_dataset(self) -> None:
        """Load and preprocess the verifiers dataset"""
        self.train_dataset = StepVerifyDataset(self.config.dataset_path).load()
        self.test_dataset = StepVerifyDataset(self.config.dataset_path).load()


    def parse_response(self, response: str) -> bool:
        """Parse model response into boolean indicating if solution is incorrect"""
        # Look for Yes/No answer, case-insensitive
        response = response.strip().lower()
        if "yes" in response:
            return True  # Solution is incorrect
        elif "no" in response:
            return False  # Solution is correct
        else:
            # Default to marking as incorrect if can't parse response
            return True

    def compute_metrics(self, predictions: List[bool], targets: List[bool]) -> Dict[str, float]:
        """Compute classification metrics"""
        targets = [self.parse_response(target) for target in targets]
        print(predictions)
        print(targets)

        return {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions),
            "recall": recall_score(targets, predictions),
            "f1": f1_score(targets, predictions, pos_label=True)
        }