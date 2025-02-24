from typing import Dict, List, Any
import re
from registry import TaskRegistry
from dataloaders.verifiers import StepVerifyDataset
from .base import Task, TaskConfig
from sklearn.metrics import f1_score


@TaskRegistry.register("mistake_location")
class MistakeLocationTask(Task):
    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_dataset(self) -> None:
        self.train_dataset = StepVerifyDataset(self.config.dataset_path).load()
        self.test_dataset = StepVerifyDataset(self.config.dataset_path).load()

    def parse_response(self, response: str) -> int:
        """Extract the step number from model response"""
        # Try to find a number in the response
        match = re.search(r'\d+', response)
        if match:
            return int(match.group())
        return 0

    def compute_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
        """Compute binary classification metrics"""
        # Convert to binary classification (error vs no error)
        predictions = [int(p) for p in predictions]
        targets = [int(t) for t in targets]
        print(predictions)
        print(targets)

        # Calculate F1 scores with different averaging methods
        f1_micro = f1_score(targets, predictions, average='micro')
        f1_macro = f1_score(targets, predictions, average='macro')
        f1_weighted = f1_score(targets, predictions, average='weighted')

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }