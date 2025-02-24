from typing import Dict, List

from mathbridge import MathBridge
from registry import TaskRegistry
from .base import Task, TaskConfig

def _is_question(text):
    """
    Check if text contains question marks or common question words
    """
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'whose', 'whom', 'can', 'could', 'would']
    text_lower = text.lower()

    return ('?' in text) or any(word in text_lower.split() for word in question_words)


@TaskRegistry.register("pedagogy_following_hard")
class PedagogyFollowingHard(Task):
    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_dataset(self) -> None:
        """Load and preprocess the verifiers dataset"""
        self.train_dataset = MathBridge(self.config.dataset_path).load()
        self.test_dataset = MathBridge(self.config.dataset_path).load()


    def parse_response(self, response: str) -> str:
        """Parse model response into boolean indicating if solution is incorrect"""
        return response

    def compute_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        return {
            "match": sum([1 for p in predictions if _is_question(p)]) / len(predictions)
        }