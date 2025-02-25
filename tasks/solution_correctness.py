from typing import Dict, List

from dataloaders.base import HuggingFaceDataset
from registry import TaskRegistry
from .base import Task, TaskConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@TaskRegistry.register("solution_correctness")
class SolutionCorrectnessTask(Task):
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

            # Create example with error
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

            # Create example without error (using reference solution)
            no_error_example = {
                'question': example['problem'],
                'student_solution': "\\n".join(["Step " + str(sub_index + 1) + " - " + substep for sub_index, substep in
                                                enumerate(example["reference_solution"].split("\n")[:-1])]),
                'is_error': False,
                'error_step': 0,  # No error
                'dialog_history': dialog_history_str,
                "student_chat_solution": example["student_correct_response"],
                "reference_solution": example["reference_solution"],
            }
            processed_examples.append(no_error_example)

        return processed_examples


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