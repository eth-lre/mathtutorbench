from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import jinja2
from dataclasses import dataclass

from dataloaders.base import HuggingFaceDataset


@dataclass
class TaskConfig:
    name: str
    dataset_path: str
    dataset_name: str
    training_split: str
    test_split: str
    system_prompt: str
    ground_truth_format: str
    few_shot_samples: Optional[List[Dict[str, Any]]] = None
    stop: Optional[str] = None


class Task(ABC):
    def __init__(self, config: TaskConfig):
        self.config = config
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from HuggingFace"""
        self.train_dataset = HuggingFaceDataset(self.config.dataset_path, self.config.dataset_name, split=self.config.training_split).load()
        self.test_dataset = HuggingFaceDataset(self.config.dataset_path, self.config.dataset_name, split=self.config.test_split).load()

    def get_system_prompt(self, example: Dict[str, Any]) -> str:
        """Render system prompt with example variables"""
        template = jinja2.Template(self.config.system_prompt)
        return template.render(**example)

    def format_ground_truth(self, example: Dict[str, Any]) -> str:
        """Format ground truth using the template"""
        template = jinja2.Template(self.config.ground_truth_format)
        return template.render(**example)

    def get_test_examples(self) -> List[Dict[str, Any]]:
        """Get test examples from the dataset"""
        return [dict(example) for example in self.test_dataset]

    def get_train_examples(self) -> List[Dict[str, Any]]:
        """Get training examples from the dataset"""
        return [dict(example) for example in self.train_dataset]

    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """Parse model response into expected format"""
        pass

    @abstractmethod
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute metrics for the task"""
        pass
